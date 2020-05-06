# Encoder

from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encodes a given batch of data.
    
    Init: 
      data_size: size of the dataset to be encoded
      z_size:    size of the z vector
      hidden_sizes: list of hidden sizes, if empty (default) then there will
                    be one layer with data_size in_features,
                    and z_size out_features
    
    Forward:
      inputs: batch of data
    
    Returns:
      Encoded data
    """
    def __init__(self, data_size, z_size, hidden_sizes=[112]):
        super(Encoder, self).__init__()

        hidden_activation = nn.ReLU()

        previous_layer_size = data_size

        layer_sizes = list(hidden_sizes) + [z_size]
        layers = []

        for layer_size in layer_sizes:
            layers.append(nn.Linear(previous_layer_size, layer_size))
            previous_layer_size = layer_size

        self.model = nn.Sequential(*layers)
        
    def forward(self, inputs):      
        return self.model(inputs)