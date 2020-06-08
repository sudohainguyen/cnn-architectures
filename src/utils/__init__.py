# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch.nn as nn


def get_activation_layer(activation, inplace=True):
    if activation == 'relu':
        return nn.ReLU(inplace=inplace)
    elif activation == 'sigmoid':
        return nn.Sigmoid(inplace=inplace)
    elif activation == 'relu6':
        return nn.ReLU6(inplace=inplace)
    else:
        raise NotImplementedError
