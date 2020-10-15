# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch.nn as nn

from ...blocks.core import Residual, Bottleneck


def _make_block(num_subblocks, in_channels, out_channels,
                bottleneck=False) -> nn.Module:
    layer = Bottleneck if bottleneck else Residual
    layers = [layer(in_channels, out_channels)]  
    
    for _ in range(num_subblocks - 1):
        layers.append(layer(out_channels, out_channels))

    return nn.Sequential(*layers)
