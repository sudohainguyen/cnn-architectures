# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn as nn

from .helpers import _make_block

cfgs = {
    18: ['basic', 2, 2, 2, 2],
    34: ['basic', 3, 4, 6, 3],
    50: ['bottleneck', 3, 4, 6, 3],
    101: ['bottleneck', 3, 4, 23, 3],
    152: ['bottleneck', 3, 8, 36, 3]
}


class ResNet(nn.Module):
    def __init__(
        self,
        n_classes,
        in_channels,
        num_layers,
        include_top=True,
        activation='softmax'
    ):
        super(ResNet, self).__init__()

        bottleneck = cfgs[num_layers][0] == 'bottleneck'
        layers = cfgs[num_layers][1:]
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # self.block_1 = _make_block(layers[0], )

        self.avgpool = nn.AvgPool2d()
        self.fc = nn.Linear()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = torch.softmax(x)
        return x
