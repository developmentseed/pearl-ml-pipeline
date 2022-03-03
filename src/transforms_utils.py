import numpy as np
import torch
import utils


def label_transforms_naip(labels, group=None):
    labels = np.array(labels).astype(np.int64)
    labels = np.where(labels == 14, 0, labels)  # to no data
    labels = np.where(labels == 15, 0, labels)  # to no data
    labels = np.where(labels == 13, 0, labels)  # to no data
    labels = np.where(labels == 10, 3, labels)  # to tree canopy
    labels = np.where(labels == 11, 3, labels)  # to tree canopy
    labels = np.where(labels == 12, 3, labels)  # to tree canopy
    labels = torch.from_numpy(labels)
    return labels


def label_transforms_epa(labels, group=None):
    labels = np.array(labels).astype(np.int64)
    labels_new = np.copy(labels)
    for k, v in utils.epa_label_dict.items():
        labels_new[labels == k] = v
    labels_new = torch.from_numpy(labels_new)
    return labels_new


def label_transform_cic(labels, group=None):
    labels = np.array(labels).astype(np.int64)
    labels_new = np.copy(labels)
    for k, v in utils.cic_label_dict.items():
        labels_new[labels == k] = v
    labels_new = torch.from_numpy(labels_new)
    return labels_new


def label_transform_naip5cls(labels, group=None):
    labels = np.array(labels).astype(np.int64)
    labels_new = np.copy(labels)
    for k, v in utils.naip_5cls.items():
        labels_new[labels == k] = v
    labels_new = torch.from_numpy(labels_new)
    return labels_new


def label_transform_4cls(labels, group=None):
    labels = np.array(labels).astype(np.int64)
    labels_new = np.copy(labels)
    for k, v in utils.naip_4cls.items():
        labels_new[labels == k] = v
    labels_new = torch.from_numpy(labels_new)
    return labels_new


def labels_transform_uvm(labels, group=None):
    labels = np.array(labels).astype(np.int64)
    labels_new = np.copy(labels)
    for k, v in utils.uvm_7cls.items():
        labels_new[labels == k] = v
    labels_new = torch.from_numpy(labels_new)
    return labels_new


def labels_transform_uvm_8cls(labels, group=None):
    labels = np.array(labels).astype(np.int64)
    labels_new = np.copy(labels)
    for k, v in utils.uvm_8cls.items():
        labels_new[labels == k] = v
    labels_new = torch.from_numpy(labels_new)
    return labels_new


def image_transforms(img, group=None):
    img = img / 255.0
    img = np.rollaxis(img, 2, 0).astype(np.float32)
    img = torch.from_numpy(img)
    return img
