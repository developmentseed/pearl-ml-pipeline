"""
Script to create seed data from trained model for PEARL MVP model retraining session
"""
import logging
import os
import sys

import joblib
import numpy as np
import sklearn.base
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import rasterio
import models

import argparse

sys.path.append("..")
LOGGER = logging.getLogger("server")

from typing import Optional, Union, List

parser = argparse.ArgumentParser(description="Create seed data from trained model")
parser.add_argument(
    "--input_csv",
    type=str,
    required=True,
    help='The path to a CSV file containing three columns -- "image_fn", "label_fn", and "group" -- that point to tiles of imagery and labels as well as which "group" each tile is in.',
)
parser.add_argument(
    "--ckpt_file", type=str, required=True, help="A trained model file in pt format"
)
parser.add_argument(
    "--n_classes", type=int, required=True, help="The number of calsses"
)
parser.add_argument(
    "--out_npz",
    type=str,
    required=True,
    help="The path to a directory to output model seed data in npz",
)
parser.add_argument(
    "--model",
    default="fcn",
    choices=("unet", "fcn", "unet2", "deeplabv3plus"),
    help="Model to use",
)

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("WARNING! Torch is reporting that CUDA isn't available, using cpu")
    device = torch.device("cpu")


def label_transforms_naip(labels):
    labels = np.array(labels).astype(np.int64)
    labels = np.where(labels == 14, 0, labels)  # to no data
    labels = np.where(labels == 15, 0, labels)  # to no data
    labels = np.where(labels == 13, 0, labels)  # to no data
    labels = np.where(labels == 10, 3, labels)  # to tree canopy
    labels = np.where(labels == 11, 3, labels)  # to tree canopy
    labels = np.where(labels == 12, 3, labels)  # to tree canopy
    return labels


def label_transforms_uvm(labels):
    naip_7cls = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
    labels = np.array(labels).astype(np.int64)
    labels_new = np.copy(labels)
    for k, v in naip_7cls.items():
        labels_new[labels == k] = v
    return labels_new


def load_model(n_classes, chkpt_file, model_nm):
    if model_nm == "unet2":
        model = models.Unet2(
            feature_scale=1,
            n_classes=n_classes,
            in_channels=4,
            is_deconv=True,
            is_batchnorm=False,
        )
    elif model_nm == "unet":
        model = models.Unet(
            feature_scale=1,
            n_classes=n_classes,
            in_channels=4,
            is_deconv=True,
            is_batchnorm=False,
        )
    elif model_nm == "fcn":
        model = models.FCN(
            num_input_channels=4,
            num_output_classes=n_classes,
            num_filters=64,
            padding=1,
        )
    elif model_nm == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=4,
            classes=n_classes,
        )
    checkpoint = torch.load(chkpt_file, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    return model


def sample_data(df_path, n_samples):

    df = pd.read_csv(df_path)
    image_fns, label_fns = df[["image_fn", "label_fn"]].values.T
    idxs = np.random.choice(image_fns.shape[0], replace=False, size=n_samples)
    image_fns = image_fns[idxs]
    label_fns = label_fns[idxs]
    return image_fns, label_fns


def deeplabv3plus_forward_features(model, x):
    features = model.encoder(x)
    decoder_output = model.decoder(*features)
    return F.interpolate(
        decoder_output,
        scale_factor=4,
    )


def get_seed_data_deeplabv3plus(
    model, device, img_fn, label_fn, n_patches, n_points, verbose=True
):

    with rasterio.open(img_fn) as f:
        data = f.read()
    data = data / 255.0

    with rasterio.open(label_fn) as f:
        labels = f.read().squeeze()

    height, width = labels.shape
    labels.shape
    labels = label_transforms_uvm(labels)

    ## Sample n_patches from the tile
    patch_size = 128
    x_imgs = np.zeros((n_patches, 4, patch_size, patch_size), dtype=np.float32)
    y_imgs = np.zeros((n_patches, patch_size, patch_size), dtype=np.uint8)
    for i in range(n_patches):

        x = np.random.randint(0, width - patch_size)
        y = np.random.randint(0, height - patch_size)

        x_img = data[:, y : y + patch_size, x : x + patch_size].copy()
        y_img = labels[y : y + patch_size, x : x + patch_size].copy()

        x_imgs[i] = x_img
        y_imgs[i] = y_img

    x_imgs = torch.from_numpy(x_imgs).to(device)
    print("x_imgs")
    print(x_imgs.shape)

    ## Run the model on the patches
    with torch.no_grad():
        x_img_features = deeplabv3plus_forward_features(model, x_imgs)
        x_img_features = x_img_features.cpu().numpy()
        print("x_img_feature_shape")
        print(x_img_features.shape)

    ## Evaluate the model on all the patches
    if verbose:
        print(
            "Base model acc on sampled patches",
            accuracy_score(y_imgs.ravel(), y_imgs_pred.ravel()),
        )
        print(
            "Base model f1 on sampled patches",
            f1_score(y_imgs.ravel(), y_imgs_pred.ravel(), average="macro"),
        )

    ## Subsample n_points from the patches
    x_seed = np.zeros((n_points, 256), dtype=np.float32)
    y_seed = np.zeros((n_points,), dtype=np.uint8)

    for j in range(n_points):
        i = np.random.randint(n_patches)
        x = np.random.randint(32, patch_size - 32)
        y = np.random.randint(32, patch_size - 32)

        x_seed[j] = x_img_features[i, :, y, x]
        y_seed[j] = y_imgs[i, y, x]

    ## Evaluate the model on the seed points
    if verbose:
        ## Use the last layer of the model to make predictions from the seed embeddings
        fcn_weights = model.last.weight.cpu().detach().numpy().squeeze()
        fcn_bias = model.last.bias.cpu().detach().numpy()
        y_seed_pred = (x_seed @ fcn_weights.T + fcn_bias).argmax(axis=1)
        print("Base model acc on subset of points", accuracy_score(y_seed, y_seed_pred))
        print(
            "Base model f1 on subset of points",
            f1_score(y_seed, y_seed_pred, average="macro"),
        )

    print("y_seed dataset for deeplabv3+ are:")
    print(y_seed.shape, np.unique(y_seed))
    return x_seed, y_seed


def get_seed_data_fcn(
    model, device, label_transform_function, img_fn, label_fn, n_patches, n_points
):
    ## Load data
    with rasterio.open(img_fn) as f:
        data = f.read()
    data = data / 255.0

    with rasterio.open(label_fn) as f:
        labels = f.read().squeeze()
    height, width = labels.shape
    labels.shape
    labels = label_transforms_function(labels)

    ## Sample n_patches from the tile
    patch_size = 256
    x_imgs = np.zeros((n_patches, 4, patch_size, patch_size), dtype=np.float32)
    y_imgs = np.zeros((n_patches, patch_size, patch_size), dtype=np.uint8)
    for i in range(n_patches):

        x = np.random.randint(0, width - patch_size)
        y = np.random.randint(0, height - patch_size)

        x_img = data[:, y : y + patch_size, x : x + patch_size].copy()
        y_img = labels[y : y + patch_size, x : x + patch_size].copy()

        x_imgs[i] = x_img
        y_imgs[i] = y_img

    x_imgs = torch.from_numpy(x_imgs).to(device)

    ## Run the model on the patches
    with torch.no_grad():
        y_imgs_pred, x_img_features = model.forward_features(x_imgs)
        y_imgs_pred = y_imgs_pred.argmax(axis=1).cpu().numpy()
        x_img_features = x_img_features.cpu().numpy()

    ## Subsample n_points from the patches
    x_seed = np.zeros((n_points, 64), dtype=np.float32)
    y_seed = np.zeros((n_points,), dtype=np.uint8)

    for j in range(n_points):
        i = np.random.randint(n_patches)
        x = np.random.randint(32, patch_size - 32)
        y = np.random.randint(32, patch_size - 32)

        x_seed[j] = x_img_features[i, :, y, x]
        y_seed[j] = y_imgs[i, y, x]

    return x_seed, y_seed


def get_seed_data_unet(
    model, device, img_fn, label_fn, n_patches, n_points, verbose=True
):

    with rasterio.open(img_fn) as f:
        data = f.read()
    data = data / 255.0

    with rasterio.open(label_fn) as f:
        labels = f.read().squeeze()

    height, width = labels.shape
    labels.shape
    labels = label_transforms_uvm(labels)

    ## Sample n_patches from the tile
    patch_size = 128
    x_imgs = np.zeros((n_patches, 4, patch_size, patch_size), dtype=np.float32)
    y_imgs = np.zeros((n_patches, patch_size, patch_size), dtype=np.uint8)
    for i in range(n_patches):

        x = np.random.randint(0, width - patch_size)
        y = np.random.randint(0, height - patch_size)

        x_img = data[:, y : y + patch_size, x : x + patch_size].copy()
        y_img = labels[y : y + patch_size, x : x + patch_size].copy()

        x_imgs[i] = x_img
        y_imgs[i] = y_img

    x_imgs = torch.from_numpy(x_imgs).to(device)
    print("x_imgs")
    print(x_imgs.shape)

    ## Run the model on the patches
    with torch.no_grad():
        x_img_features = model.forward_features(x_imgs)
        # y_imgs_pred = y_imgs_pred.argmax(axis=1).cpu().numpy()
        x_img_features = x_img_features.cpu().numpy()
        print("x_img_features shape")
        print(x_img_features.shape)

    ## Evaluate the model on all the patches
    if verbose:
        print(
            "Base model acc on sampled patches",
            accuracy_score(y_imgs.ravel(), y_imgs_pred.ravel()),
        )
        print(
            "Base model f1 on sampled patches",
            f1_score(y_imgs.ravel(), y_imgs_pred.ravel(), average="macro"),
        )

    ## Subsample n_points from the patches
    x_seed = np.zeros((n_points, 64), dtype=np.float32)
    y_seed = np.zeros((n_points,), dtype=np.uint8)

    for j in range(n_points):
        i = np.random.randint(n_patches)
        x = np.random.randint(32, patch_size - 32)
        y = np.random.randint(32, patch_size - 32)

        x_seed[j] = x_img_features[i, :, y, x]
        y_seed[j] = y_imgs[i, y, x]

    ## Evaluate the model on the seed points
    if verbose:
        ## Use the last layer of the model to make predictions from the seed embeddings
        fcn_weights = model.last.weight.cpu().detach().numpy().squeeze()
        fcn_bias = model.last.bias.cpu().detach().numpy()
        y_seed_pred = (x_seed @ fcn_weights.T + fcn_bias).argmax(axis=1)
        print("Base model acc on subset of points", accuracy_score(y_seed, y_seed_pred))
        print(
            "Base model f1 on subset of points",
            f1_score(y_seed, y_seed_pred, average="macro"),
        )

    return x_seed, y_seed


def calculate_seed_data():
    device = torch.device("cuda")
    x_test = []
    y_test = []

    df = pd.read_csv(args.input_csv)
    image_fns, label_fns = df[["image_fn", "label_fn"]].values.T
    for i in range(len(image_fns)):
        if i % 5 == 0:
            print(i, len(image_fns))

        # -------------------
        # Setup model
        # -------------------
        if args.model == "unet2":
            model = load_model(args.n_classes, args.ckpt_file, args.model)
            x_test_sample, y_test_sample = get_seed_data_unet(
                model,
                device,
                image_fns[i],
                label_fns[i],
                n_patches=128,
                n_points=1000,
                verbose=False,
            )
            x_test.append(x_test_sample)
            y_test.append(y_test_sample)
        elif args.model == "unet":
            model = model = load_model(args.n_classes, args.ckpt_file, args.model)
            x_test_sample, y_test_sample = get_seed_data_unet(
                model,
                device,
                image_fns[i],
                label_fns[i],
                n_patches=128,
                n_points=1000,
                verbose=False,
            )
            x_test.append(x_test_sample)
            y_test.append(y_test_sample)
        elif args.model == "fcn":
            model = model = load_model(args.n_classes, args.ckpt_file, args.model)
            x_test_sample, y_test_sample = get_seed_data_fcn(
                model,
                device,
                image_fns[i],
                label_fns[i],
                n_patches=64,
                n_points=100,
                verbose=False,
            )
            x_test.append(x_test_sample)
            y_test.append(y_test_sample)
        elif args.model == "deeplabv3plus":
            model = model = load_model(args.n_classes, args.ckpt_file, args.model)
            x_test_sample, y_test_sample = get_seed_data_deeplabv3plus(
                model,
                device,
                image_fns[i],
                label_fns[i],
                n_patches=64,
                n_points=100,
                verbose=False,
            )
            x_test.append(x_test_sample)
            y_test.append(y_test_sample)
        else:
            raise ValueError("Invalid model")

    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    print(np.unique(y_test))

    np.savez(args.out_npz, embeddings=x_test, labels=y_test)


if __name__ == "__main__":
    calculate_seed_data()
