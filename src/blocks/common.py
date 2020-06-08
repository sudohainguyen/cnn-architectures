# Copyright (c) 2020 Hai Nguyen
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from .core import Conv2D


def conv2d_3x3(in_channels, out_channels, use_bn=True, strides=1, groups=1):
    return Conv2D(in_channels, out_channels, kernel_size=3, use_bn=use_bn,
                  strides=strides, groups=groups)


def conv2d_1x1(in_channels, out_channels, use_bn=True, strides=1, groups=1):
    return Conv2D(in_channels, out_channels, kernel_size=1, use_bn=use_bn,
                  strides=strides, groups=groups)
