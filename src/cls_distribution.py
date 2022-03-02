import sys
import os

os.environ[
    "CURL_CA_BUNDLE"
] = "/etc/ssl/certs/ca-certificates.crt"  # A workaround in case this happens: https://github.com/mapbox/rasterio/issues/1289
import time
import datetime
import argparse
import copy

import numpy as np
import pandas as pd
import json
import utils

import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioError, RasterioIOError

from transforms_utils import (
    labels_transform_uvm_8cls,
)

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

import seaborn as sns

sns.set()

import torch

NUM_WORKERS = 6
CHIP_SIZE = 256

parser = argparse.ArgumentParser(
    description="Minic streaming label data to create class distribution"
)
parser.add_argument(
    "--input_fn",
    type=str,
    required=True,
    help='The path to a CSV file containing three columns -- "image_fn", "label_fn", and "group" -- that point to tiles of imagery and labels as well as which "group" each tile is in.',
)
parser.add_argument(
    "--label_transform",
    default="naip",
    required=True,
    help="str either naip or epa to indicate how to transform labels",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="The path to store class distribution.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Flag for overwriting `output_dir` if that directory already exists.",
)
## Training arguments to generate class distribution
parser.add_argument(
    "--batch_size", type=int, default=16, help="Batch size to use for training"
)
parser.add_argument(
    "--num_epochs", type=int, default=1, help="Number of epochs to train for"
)
parser.add_argument(
    "--seed", type=int, default=0, help="Random seed to pass to numpy and torch"
)
parser.add_argument(
    "--num_classes", type=int, default=10, help="number of classes in dataset"
)
parser.add_argument(
    "--num_chips",
    type=int,
    default=40,
    help="number of chips to randomly sample from data",
)
args = parser.parse_args()


def stream_tile_fns(NUM_WORKERS, label_fns, groups):
    worker_info = torch.utils.data.get_worker_info()
    if (
        worker_info is None
    ):  # In this case we are not loading through a DataLoader with multiple workers
        worker_id = 0
        num_workers = 1
    else:
        worker_id = worker_info.id
        num_workers = worker_info.NUM_WORKERS

    # We only want to shuffle the order we traverse the files if we are the first worker (else, every worker will shuffle the files...)
    if worker_id == 0:
        np.random.shuffle(label_fns)  # in place
    # This logic splits up the list of filenames into `num_workers` chunks. Each worker will recieve ceil(num_filenames / num_workers) filenames to generate chips from. If the number of workers doesn't divide the number of filenames evenly then the last worker will have fewer filenames.
    N = len(label_fns)
    num_files_per_worker = int(np.ceil(N / num_workers))
    lower_idx = worker_id * num_files_per_worker
    upper_idx = min(N, (worker_id + 1) * num_files_per_worker)
    for idx in range(lower_idx, upper_idx):

        label_fn = None
        # if self.use_labels:
        label_fn = label_fns[idx]
        group = groups[idx]
        print(label_fn)

        yield label_fn, group


def stream_chips(
    num_workers,
    label_fns,
    num_chips_per_tile,
    groups,
    CHIP_SIZE,
    windowed_sampling,
    nodata_check,
    label_transform,
    verbose,
):
    for label_fn, group in stream_tile_fns(num_workers, label_fns, groups):
        num_skipped_chips = 0
        # Open file pointers
        label_fp = rasterio.open(label_fn, "r")

        # if use_labels: # garuntee that our label mask has the same dimensions as our imagery
        t_height, t_width = label_fp.shape
        print("Height and width of the label are:")
        print(t_height, t_width)

        # If we aren't in windowed sampling mode then we should read the entire tile up front
        label_data = None
        try:
            if not windowed_sampling:
                label_data = (
                    label_fp.read().squeeze()
                )  # assume the label geotiff has a single channel
        except RasterioError as e:
            print("WARNING: Error reading in entire file, skipping to the next file")
            continue

        for i in range(num_chips_per_tile):
            # Select the top left pixel of our chip randomly
            x = np.random.randint(0, t_width - CHIP_SIZE)
            y = np.random.randint(0, t_height - CHIP_SIZE)

            # Read labels
            labels = None
            if windowed_sampling:
                try:
                    labels = label_fp.read(
                        window=Window(x, y, CHIP_SIZE, CHIP_SIZE)
                    ).squeeze()
                except RasterioError:
                    print(
                        "WARNING: Error reading chip from file, skipping to the next chip"
                    )
                    continue
            else:
                labels = label_data[y : y + CHIP_SIZE, x : x + CHIP_SIZE]

            # # Check for no data
            if nodata_check is not None:
                skip_chip = nodata_check(labels)

                if (
                    skip_chip
                ):  # The current chip has been identified as invalid by the `nodata_check(...)` method
                    num_skipped_chips += 1
                    continue
            if label_transform is not None:
                labels = label_transform(labels, group)
            else:
                labels = torch.from_numpy(labels).squeeze()
            print(labels)
            return labels
        label_fp.close()
        #
        if num_skipped_chips > 0 and verbose:
            print("We skipped %d chips on %s" % (label_fn))


def label_transforms_naip(labels, group):
    labels = np.array(labels).astype(np.int64)
    labels = np.where(labels == 14, 0, labels)  # to no data
    labels = np.where(labels == 15, 0, labels)  # to no data
    labels = np.where(labels == 13, 0, labels)  # to no data
    labels = np.where(labels == 10, 3, labels)  # to tree canopy
    labels = np.where(labels == 11, 3, labels)  # to tree canopy
    labels = np.where(labels == 12, 3, labels)  # to tree canopy
    return labels


def label_transforms_epa(labels, group):
    labels = np.array(labels).astype(np.int64)
    labels_new = np.copy(labels)
    for k, v in utils.epa_label_dict.items():
        labels_new[labels == k] = v
    labels_new = torch.from_numpy(labels_new)
    return labels_new


def label_transforms_uvm(labels, group):
    labels = np.array(labels).astype(np.int64)
    labels_new = np.copy(labels)
    for k, v in utils.uvm_7cls.items():
        labels_new[labels == k] = v
    labels_new = torch.from_numpy(labels_new)
    return labels_new


def nodata_check(labels):
    return np.any(labels == 0)


def class_distribute():
    print(
        "Starting DFC2021 baseline training script at %s"
        % (str(datetime.datetime.now()))
    )
    num_chips_per_tile = args.num_chips
    windowed_sampling = False
    label_transform = args.label_transform
    nodata_check = None
    verbose = True
    all_labels = []

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
    elif args.label_transform == "uvm":
        label_transform = label_transforms_uvm
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
        ]
    else:
        raise ValueError("Invalid label transform")
    # -------------------
    # Setup
    # -------------------
    assert os.path.exists(args.input_fn)

    if os.path.isfile(args.output_dir):
        print("A file was passed as `--output_dir`, please pass a directory!")
        return

    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)):
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
        print("WARNING! Torch is reporting that CUDA isn't available, using cpu")
        device = "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -------------------
    # Load input data
    # -------------------
    input_dataframe = pd.read_csv(args.input_fn)
    print(input_dataframe.head())
    label_fns = input_dataframe["label_fn"].values
    groups = input_dataframe["group"].values
    print(label_fns)

    print(args.label_transform)
    if args.label_transform == "naip":
        label_transform = label_transforms_naip
    elif args.label_transform == "epa":
        label_transform = label_transforms_epa
    elif args.label_transform == "uvm":
        label_transform = label_transforms_uvm
    else:
        raise ValueError("Invalid label transform")

    num_training_batches_per_epoch = int(
        len(label_fns) * args.num_chips / args.batch_size
    )

    # getting label chips stac by given model epochs
    # is num_chips_per_tile the num_training_batches_per_epoch
    for epoch in range(args.num_epochs):
        for num_batches in range(num_training_batches_per_epoch):
            try:
                labels = stream_chips(
                    NUM_WORKERS,
                    label_fns,
                    num_chips_per_tile,
                    groups,
                    CHIP_SIZE,
                    windowed_sampling,
                    nodata_check,
                    label_transform,
                    verbose,
                )
                all_labels.append(labels)
            except Exception as ex:
                print(ex)
                pass
    label_arr = np.array([t.numpy() for t in all_labels])

    # -------------------
    # Plot classes distribution
    # -------------------
    fig, ax = plt.subplots(figsize=(30, 10))
    fig.tight_layout()
    unique, counts = np.unique(label_arr, return_counts=True)
    cls_dict = dict(zip(range(len(class_names)), class_names))
    vc_out = dict(zip(unique, counts / args.num_epochs))  # 10 epoaches
    vc2df = dict(zip(cls_dict.values(), vc_out.values()))
    df = pd.DataFrame.from_dict(vc2df, orient="index", columns=["count"])

    ax = sns.barplot(x=df.index, y="count", data=df)
    ax.set_ylabel(f"Class count")
    ax.set_xlabel(f"Classe name")
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "cls_distribution.png"))
    csv_fn = "output_cls_counts_and_values.csv"
    df.to_csv(os.path.join(args.output_dir, csv_fn))


if __name__ == "__main__":
    class_distribute()
