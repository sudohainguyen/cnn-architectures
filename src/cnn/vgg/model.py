# Copyright (c) 2020 Hai Nguyen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn as nn

from .helpers import _make_blocks
from ...utils import get_activation_layer


configs = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M',
         512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
         512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
         512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):
    def __init__(
        self,
        n_classes,
        in_channels,
        num_layers,
        use_batchnorm=False,
        include_top=True,
        activation='softmax'
    ):
        """Base model of VGG

        Parameters
        ----------
        n_classes : int
            Number of classes
        in_channels : int
            Input channels
        num_layers : int
            Number of NN layers, corresponds with VGG version
        use_batchnorm : bool, optional
            Apply BatchNormalization, by default False
        include_top : bool, optional
            Include prediction output, by default True
        activation : str, optional
            Activation function name, by default 'softmax'
        """    
        super(VGG, self).__init__()
        self.features = _make_blocks(configs[num_layers],
                                     in_channels, use_batchnorm)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        if include_top:
            self.classifier.add(
                nn.Linear(4096, n_classes),
                get_activation_layer(activation)
            )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def vgg16(n_classes, in_channels, use_batchnorm=False,
          include_top=True, activation='softmax'):
    return VGG(n_classes, in_channels, 16, use_batchnorm,
               include_top, activation)


def vgg19(n_classes, in_channels, use_batchnorm=False,
          include_top=True, activation='softmax'):
    return VGG(n_classes, in_channels, 19, use_batchnorm,
               include_top, activation)
