import sys
import os
import time
import datetime
import argparse

import numpy as np
import pandas as pd

import rasterio
from rasterio.windows import Window

import torch
import torch.nn.functional as F

import models
from dataloaders.TileDatasets import TileInferenceDataset
import utils
from sklearn.metrics import confusion_matrix, f1_score

os.environ[
    "CURL_CA_BUNDLE"
] = "/etc/ssl/certs/ca-certificates.crt"  # A workaround in case this happens: https://github.com/mapbox/rasterio/issues/1289

NUM_WORKERS = 4
CHIP_SIZE = 256
PADDING = 128
assert PADDING % 2 == 0
HALF_PADDING = PADDING // 2
CHIP_STRIDE = CHIP_SIZE - PADDING

from azureml.core import Run

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
    "--model", default="fcn", choices=("unet", "fcn"), help="Model to use"
)

parser.add_argument(
    "--num_classes",
    type=int,
    default=11,
    help="number of classes model was trained with",
),

parser.add_argument(
    "--label_transform",
    default="naip",
    help="str either naip, epa or cic to indicate how to transform labels",
)


args = parser.parse_args()


def label_transforms_naip(labels):
    labels = np.array(labels).astype(np.int64)
    labels = np.where(labels == 14, 0, labels)  # to no data
    labels = np.where(labels == 15, 0, labels)  # to no data
    labels = np.where(labels == 13, 0, labels)  # to no data
    labels = np.where(labels == 10, 3, labels)  # to tree canopy
    labels = np.where(labels == 11, 3, labels)  # to tree canopy
    labels = np.where(labels == 12, 3, labels)  # to tree canopy
    return labels


def label_transforms_epa(labels):
    labels = np.array(labels).astype(np.int64)
    labels_new = np.copy(labels)
    for k, v in utils.epa_label_dict.items():
        labels_new[labels == k] = v
    return labels_new


def label_transform_cic(labels):
    labels = np.array(labels).astype(np.int64)
    labels_new = np.copy(labels)
    for k, v in utils.cic_label_dict.items():
        labels_new[labels == k] = v
    return labels_new


def random_pixel_values(src_path: str, number_of_point: int, excludes={10, 11, 12}):
    with rasterio.open(src_path) as src_dst:
        output = {}
        cr_lst = []
        arr = src_dst.read(indexes=1)
        value, count = np.unique(arr, return_counts=True)
        for (i, c) in enumerate(count):
            if value[i] in excludes:
                continue
            point_y, point_x = np.where(arr == value[i])
            n_points = (
                number_of_point if len(point_x) > number_of_point else len(point_x)
            )
            indexes = np.random.choice(len(point_x), n_points).tolist()
            # y,x in row/col indexes
            cr = [(point_x[idx], point_y[idx]) for idx in indexes]
            output[value[i]] = cr
            cr_lst.append(cr)
            # TODO
            # yield pix, coordinates
    cr_f = [list(item) for sublist in cr_lst for item in sublist]
    return output, cr_f


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
    else:
        raise ValueError("Invalid label transform")

    # -------------------
    # Run on each line in the input
    # -------------------
    input_dataframe = pd.read_csv(args.input_fn)
    image_fns = input_dataframe["image_fn"].values
    label_fns = input_dataframe["label_fn"].values
    groups = input_dataframe["group"].values

    # Get Row,Column for unique lables
    for i, gt_img in enumerate(label_fns):
        x_embedding = []
        output, cr_f = random_pixel_values(gt_img, 10, {10, 11, 12})
        labels = list(output.keys()) * 10
        labels.sort()
        # print(output)
        print("flattened coords:")
        print(cr_f)
        print("labels")
        print(labels)

        # run inference on window that contains each row,column val
        for rc in cr_f:
            gt_2 = rasterio.open(image_fns[i])
            w = gt_2.read(window=Window(rc[0], rc[1], 256, 256))
            data = w / 255.0
            data = data.astype(np.float32)
            data = torch.from_numpy(data)
            data = data.to(device)

            label_img = rasterio.open(gt_img)
            w_label = label_img.read(1, window=Window(rc[0], rc[1], 256, 256))
            print(w_label[0, 0])

            with torch.no_grad():

                embedding = model.forward_features(
                    data[None, ...]
                )  # insert singleton "batch" dimension to input data for pytorch to be happy
                embedding = embedding.cpu().numpy()
                embedding = np.moveaxis(embedding[0], 0, -1)
                x_embedding.append(embedding[0, 0])

        output_fn = gt_img[0][:-4].split("/")[-1]  # something like "546_naip-2013.tif"
        output_fn_e = output_fn + "_embedding.npz"
        output_fn_l = output_fn + "_label.npz"

        output_path_e = os.path.join(args.output_dir, output_fn_e)
        output_path_label = os.path.join(args.output_dir, output_fn_l)

        np.savez(output_path_e, np.array(x_embedding))
        np.savez(output_path_label, np.array(labels))
        print("saved")


if __name__ == "__main__":
    main()
