# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch.nn as nn
from ..utils import get_activation_layer


class Conv2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        use_pad=True,
        use_bn=True,
        bn_eps=1e-5,
        activation='relu'
    ):
        super(Conv2D, self).__init__()
        
        self.use_bn = use_bn
        self.use_pad = use_pad

        if use_pad:
            self.pad = nn.ZeroPad2d(padding)
            padding = 0

        self.core = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=(not use_bn),
            groups=groups)

        if use_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        
        if activation:
            self.activation = get_activation_layer(activation)

    def forward(self, x):
        if self.use_pad:
            x = self.pad(x)

        x = self.core(x)
        
        if self.use_bn:
            x = self.bn(x)
        
        if self.activation:
            x = self.activation(x)

        return x
