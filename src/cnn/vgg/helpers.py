# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch.nn as nn

from ...blocks import conv2d_3x3


def _make_blocks(cfg: list, in_channels=3, use_batchnorm=False) -> nn.Module:
    """Generate a set of layers for a vgg block

    Parameters
    ----------
    cfg : list
        Config list
    in_channels : int, optional
        Input channels, by default 3
    use_batchnorm : bool, optional
        Apply BatchNormalization in Conv, by default False

    Returns
    -------
    nn.Module
        A VGG Block
    """    
    layers = []
    for item in cfg:
        if item == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2))
            continue
        layers.append(conv2d_3x3(in_channels, item, use_bn=use_batchnorm))
        in_channels = item

    return nn.Sequential(*layers)
