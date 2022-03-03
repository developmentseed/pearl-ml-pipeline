import os

import time
import datetime
import argparse

import numpy as np
import pandas as pd

import rasterio

import torch
import torch.nn.functional as F

import models
from dataloaders.TileDatasets import TileInferenceDataset
import utils
from transforms_utils import (
    label_transforms_naip,
    label_transform_cic,
    label_transforms_epa,
    label_transform_naip5cls,
    labels_transform_uvm,
    labels_transform_uvm_8cls,
    image_transforms,
)
from sklearn.metrics import f1_score
from azureml.core import Run

# A workaround in case this happens: https://github.com/mapbox/rasterio/issues/1289
os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"

NUM_WORKERS = 4
CHIP_SIZE = 256
PADDING = 128
assert PADDING % 2 == 0
HALF_PADDING = PADDING // 2
CHIP_STRIDE = CHIP_SIZE - PADDING


run = Run.get_context()

parser = argparse.ArgumentParser(description="DFC2021 model inference script")
parser.add_argument(
    "--input_fn",
    type=str,
    required=True,
    help='The path to a CSV file containing three columns -- "image_fn", "label_fn", and "group" -- that point to tiles of imagery and labels as well as which "group" each tile is in.',
)
parser.add_argument(
    "--model_fn", type=str, required=True, help="Path to the model file to use."
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="The path to output the model predictions as a GeoTIFF. Will fail if this file already exists.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Flag for overwriting `--output_dir` if that directory already exists.",
)
parser.add_argument("--gpu", type=int, default=0, help="The ID of the GPU to use")
parser.add_argument(
    "--batch_size", type=int, default=2, help="Batch size to use during inference."
)
parser.add_argument(
    "--save_soft",
    action="store_true",
    help='Flag that enables saving the predicted per class probabilities in addition to the "hard" class predictions.',
)
parser.add_argument(
    "--model",
    default="fcn",
    choices=("unet", "fcn", "unet2", "deeplabv3plus"),
    help="Model to use",
)

parser.add_argument(
    "--num_classes",
    type=int,
    default=10,
    help="number of classes model was trained with",
),

parser.add_argument(
    "--label_transform",
    default="naip",
    help="str either naip, epa or cic to indicate how to transform labels",
)


args = parser.parse_args()


def main():
    print("Starting model eval script at %s" % (str(datetime.datetime.now())))

    # -------------------
    # Setup
    # -------------------
    assert os.path.exists(args.input_fn)
    assert os.path.exists(args.model_fn)

    if os.path.isfile(args.output_dir):
        print("A file was passed as `--output_dir`, please pass a directory!")
        return

    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        if args.overwrite:
            print(
                "WARNING! The output directory, %s, already exists, we might overwrite data in it!"
                % (args.output_dir)
            )
        else:
            print(
                "The output directory, %s, already exists and isn't empty. We don't want to overwrite and existing results, exiting..."
                % (args.output_dir)
            )
            return
    else:
        print("The output directory doesn't exist or is empty.")
        os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % args.gpu)
    else:
        print("WARNING! Torch is reporting that CUDA isn't available, exiting...")
        return

    # -------------------
    # Load model
    # -------------------
    if args.model == "unet":
        model = models.get_unet(classes=args.num_classes)
    elif args.model == "fcn":
        model = models.get_fcn(num_output_classes=args.num_classes)
    elif args.model == "unet2":
        model = models.get_unet2(n_classes=args.num_classes)
    elif args.model == "deeplabv3plus":
        model = models.get_deeplabv3plus(n_classes=args.num_classes)
    else:
        raise ValueError("Invalid model")
    model.load_state_dict(torch.load(args.model_fn))
    model = model.to(device)

    # determine which label transform to use
    if args.label_transform == "naip":
        label_transform = label_transforms_naip
        class_names = [
            "no_data",
            "water",
            "emergent_wetlands",
            "tree_canopy",
            "shrubland",
            "low_vegetation",
            "barren",
            "structure",
            "impervious_surface",
            "impervious_roads",
            "weighted_avg",
        ]
    elif args.label_transform == "epa":
        label_transform = label_transforms_epa
        class_names = [
            "no_data",
            "impervious",
            "soil_barren",
            "grass",
            "tree/forest",
            "water",
            "shrub",
            "woody_wetlands",
            "emergent_wetlands",
            "agriculture",
            "orchard",
            "weighted_avg",
        ]
    elif args.label_transform == "cic":
        label_transform = label_transform_cic
        class_names = [
            "Structures",
            "Impervious Surface",
            "Water",
            "Grassland/Pairie",
            "Tree Canopy",
            "Turff",
            "Barren/Rock",
            "Irregated",
        ]

    elif args.label_transform == "naip_5cls":
        label_transform = label_transform_naip5cls
        class_names = [
            "water/wetland",
            "tree",
            "barren",
            "low veg",
            "built enviornment",
        ]

    elif args.label_transform == "naip_4cls":
        label_transform = label_transform_naip5cls
        class_names = ["water/wetland", "tree", "low veg", "built enviornment"]
    elif args.label_transform == "uvm":
        label_transform = labels_transform_uvm
        class_names = [
            "tree",
            "grass",
            "bare soil",
            "water",
            "buildings",
            "roads",
            "other impervious",
        ]
    elif args.label_transform == "uvm8cls":
        label_transform = labels_transform_uvm_8cls
        class_names = [
            "tree",
            "grass",
            "bare soil",
            "water",
            "buildings",
            "roads",
            "other impervious",
            "shrubs",
        ]

    else:
        raise ValueError("Invalid label transform")

    # -------------------
    # Run on each line in the input
    # -------------------
    input_dataframe = pd.read_csv(args.input_fn)
    image_fns = input_dataframe["image_fn"].values
    label_fns = input_dataframe["label_fn"].values

    df_lst = []
    for image_idx in range(len(image_fns)):
        pred_masks = []
        tic = time.time()
        image_fn = image_fns[image_idx]
        gt_label_fn = label_fns[image_idx]

        print(
            "(%d/%d) Processing %s" % (image_idx, len(image_fns), image_fn), end=" ... "
        )

        # -------------------
        # Load input and create dataloader
        # -------------------

        with rasterio.open(image_fn) as f:
            input_width, input_height = f.width, f.height
            input_profile = f.profile.copy()

        dataset = TileInferenceDataset(
            image_fn,
            chip_size=CHIP_SIZE,
            stride=CHIP_STRIDE,
            transform=image_transforms,
            verbose=False,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # -------------------
        # Run model and organize output
        # -------------------

        output = np.zeros(
            (args.num_classes, input_height, input_width), dtype=np.float32
        )
        kernel = np.ones((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
        kernel[HALF_PADDING:-HALF_PADDING, HALF_PADDING:-HALF_PADDING] = 5
        counts = np.zeros((input_height, input_width), dtype=np.float32)

        for i, (data, coords) in enumerate(dataloader):
            data = data.to(device)
            with torch.no_grad():
                # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
                model.eval()
                t_output = model(data)
                t_output = F.softmax(t_output, dim=1).cpu().numpy()

            for j in range(t_output.shape[0]):
                y, x = coords[j]

                output[:, y : y + CHIP_SIZE, x : x + CHIP_SIZE] += t_output[j] * kernel
                counts[y : y + CHIP_SIZE, x : x + CHIP_SIZE] += kernel

        output = output / counts
        output_hard = output.argmax(axis=0).astype(np.uint8)

        # append to list of preds
        pred_masks.append(output_hard)

        # -------------------
        # Save output
        # -------------------
        output_profile = input_profile.copy()
        output_profile["driver"] = "GTiff"
        output_profile["dtype"] = "uint8"
        output_profile["count"] = 1
        output_profile["nodata"] = 90

        output_fn = image_fn.split("/")[-1]  # something like "546_naip-2013.tif"
        output_fn = output_fn.replace("naip", "predictions")
        output_fn = os.path.join(args.output_dir, output_fn)

        with rasterio.open(output_fn, "w", **output_profile) as f:
            f.write(output_hard, 1)
            f.write_colormap(1, utils.LC_TREE_COLORMAP)  # fix

        if args.save_soft:

            output = output / output.sum(axis=0, keepdims=True)
            output = (output * 255).astype(np.uint8)

            output_profile = input_profile.copy()
            output_profile["driver"] = "GTiff"
            output_profile["dtype"] = "uint8"
            output_profile["count"] = 13
            # output_profile["count"] = 13
            del output_profile["nodata"]

            output_fn = image_fn.split("/")[-1]  # something like "546_naip-2013.tif"
            output_fn = output_fn.replace("naip", "predictions-soft")
            output_fn = os.path.join(args.output_dir, output_fn)

            with rasterio.open(output_fn, "w", **output_profile) as f:
                f.write(output)

        print("finished in %0.4f seconds" % (time.time() - tic))

        # load in ground truth
        gt = rasterio.open(gt_label_fn).read()
        gt = gt[0]
        gt_f = np.reshape(gt, [-1])

        # remove no data vals
        gt_cleaned = np.delete(
            gt_f, np.where((gt_f == 15) | (gt_f == 14) | (gt_f == 13) | (gt_f == 0))
        )

        print(gt_cleaned.shape)
        print(np.unique(gt_cleaned))

        gt_t = label_transform(gt_cleaned)
        print("label transformed unique")
        print(np.unique(gt_t))

        # f-score calculation
        pred_masks = np.array(pred_masks)
        pred_masks_f = np.reshape(pred_masks, [-1])

        pred_masks_cleaned = np.delete(
            pred_masks_f,
            np.where((gt_f == 15) | (gt_f == 14) | (gt_f == 13) | (gt_f == 0)),
        )
        print(pred_masks_cleaned.shape)

        uniq_tm = np.unique(gt_t)  # unique true mask
        print(uniq_tm)
        uniq_pm = np.unique(pred_masks_cleaned)  # unique pred mask

        # f1 score is computed toward common classes between gt and pred
        uniq_v = np.unique(np.concatenate((uniq_tm, uniq_pm)))
        print(uniq_v)

        # determine missing labels
        missing_labels = np.setdiff1d(list(np.arange(args.num_classes)), uniq_v)

        f1_score_weighted = f1_score(gt_t, pred_masks_cleaned, average="weighted")

        f1_score_per_class = f1_score(gt_t, pred_masks_cleaned, average=None)
        print(f"Length of f1 for classes {len(f1_score_per_class)}, they are: \n")
        print(f1_score_per_class)

        per_class_f1_final = np.zeros(len(class_names))
        # where the unique cls id exist, fill in f1 per calss
        per_class_f1_final[uniq_v] = f1_score_per_class
        # where is the missing id, fill in np.nan
        per_class_f1_final[missing_labels] = np.nan
        per_class_f1_final[-1] = f1_score_weighted

        d = {
            "class": class_names,
            image_fn.split("/")[-1] + "_f1_score": per_class_f1_final,
        }

        df = pd.DataFrame.from_dict(d)
        df_t = df.T
        df_t.columns = df["class"]
        df_t = df_t.drop("class")

        df_lst.append(df_t)

    df_combine = pd.concat(df_lst)
    df_combine.loc["mean"] = df_combine.mean(axis=0)
    csv_fn = os.path.join(args.output_dir, "f1_score_stats.csv")
    df_combine.to_csv(csv_fn)


if __name__ == "__main__":
    main()
