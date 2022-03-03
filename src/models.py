import functools
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp

import utils

from typing import Optional, Union, List


class FCN(nn.Module):
    def __init__(self, num_input_channels, num_output_classes, num_filters=64):
        super(FCN, self).__init__()

        self.conv1 = nn.Conv2d(
            num_input_channels, num_filters, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        self.last = nn.Conv2d(
            num_filters, num_output_classes, kernel_size=1, stride=1, padding=0
        )

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.last(x)
        return x

    def forward_features(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        z = F.relu(self.conv5(x))
        # y = self.last(z)
        return z


class Unet(smp.base.SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.
    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: Unet
    .. _Unet:
        https://arxiv.org/abs/1505.04597
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = smp.unet.decoder.UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=1,
        )

        if aux_params is not None:
            self.classification_head = smp.base.ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()


class Unet2(nn.Module):
    def __init__(
        self,
        feature_scale=1,
        n_classes=3,
        in_channels=3,
        is_deconv=True,
        is_batchnorm=False,
    ):
        """
        Args:
            feature_scale: the smallest number of filters (depth c) is 64 when feature_scale is 1,
                           and it is 32 when feature_scale is 2
            n_classes: number of output classes
            in_channels: number of channels in input
            is_deconv:
            is_batchnorm:
        """

        super(Unet2, self).__init__()

        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        assert (
            64 % self.feature_scale == 0
        ), f"feature_scale {self.feature_scale} does not work with this UNet"

        filters = [
            64,
            128,
            256,
            512,
            1024,
        ]  # this is `c` in the diagram, [c, 2c, 4c, 8c, 16c]
        filters = [int(x / self.feature_scale) for x in filters]
        logging.info("filters used are: {}".format(filters))

        # downsampling
        self.conv1 = UnetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UnetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UnetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UnetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = UnetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UnetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UnetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UnetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

    def forward_features(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final, up1


class UnetConv2(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm):
        super(UnetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                # this amount of padding/stride/kernel_size preserves width/height
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, stride=1, padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                ),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, stride=1, padding=1
                ),
                nn.ReLU(),
            )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, is_deconv):
        """
        is_deconv:  use transposed conv layer to upsample - parameters are learnt; otherwise use
                    bilinear interpolation to upsample.
        """
        super(UnetUp, self).__init__()

        self.conv = UnetConv2(in_channels, out_channels, False)

        self.is_deconv = is_deconv
        if is_deconv:
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        # UpsamplingBilinear2d is deprecated in favor of interpolate()
        # else:
        #     self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        """
        inputs1 is from the downward path, of higher resolution
        inputs2 is from the 'lower' layer. It gets upsampled (spatial size increases) and its depth (channels) halves
        to match the depth of inputs1, before being concatenated in the depth dimension.
        """
        if self.is_deconv:
            outputs2 = self.up(inputs2)
        else:
            # scale_factor is the multiplier for spatial size
            outputs2 = F.interpolate(
                inputs2, scale_factor=2, mode="bilinear", align_corners=True
            )

        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)

        return self.conv(torch.cat([outputs1, outputs2], dim=1))


def get_unet(classes=11):
    return Unet(
        encoder_name="resnet18",
        encoder_depth=3,
        encoder_weights=None,
        decoder_channels=(128, 64, 64),
        in_channels=4,
        classes=classes,
    )


def get_unet2(n_classes):
    return Unet2(
        feature_scale=1,
        n_classes=n_classes,
        in_channels=4,
        is_deconv=True,
        is_batchnorm=False,
    )


def get_fcn(num_output_classes=11):
    return FCN(
        num_input_channels=4, num_output_classes=num_output_classes, num_filters=64
    )


def get_deeplabv3plus(n_classes):
    return smp.DeepLabV3Plus(
        encoder_name="resnet18",
        in_channels=4,
        classes=n_classes,
        encoder_weights=None,
    )
