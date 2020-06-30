# -*- coding: utf-8 -*-
"""
The majority of this script comes straight from https://github.com/rcamino/multi-categorical-gans
"""

from __future__ import print_function

import torch.nn as nn

from gan_scripts.singleoutput import SingleOutput
from gan_scripts.multioutput import MultiCategorical

class Generator2(nn.Module):

    def __init__(self,
                 noise_size,          # Size of z vector
                 output_size,         # Size of z vector plus number of continous variables
                 hidden_sizes=[],     # A list of hidden layer sizes
                 bn_decay=0.01        # batch norm decay parameter
                 ):
        super(Generator2, self).__init__()

        hidden_activation = nn.ReLU()

        previous_layer_size = noise_size
        hidden_layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))
            if layer_number > 0 and bn_decay > 0:
                hidden_layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))
            hidden_layers.append(hidden_activation)
            previous_layer_size = layer_size

        if len(hidden_layers) > 0:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        else:
            self.hidden_layers = None
        
        # Define size of final output layer
        if type(output_size) is int:
            self.output = SingleOutput(previous_layer_size, output_size)
        elif type(output_size) is list:
            self.output = MultiCategorical(previous_layer_size, output_size)
        else:
            raise Exception("Invalid output size.")


    def forward(self, noise, training=False, temperature=None):
        if self.hidden_layers is None:
            hidden = noise
        else:
            hidden = self.hidden_layers(noise)

        return self.output(hidden, training=training, temperature=temperature)