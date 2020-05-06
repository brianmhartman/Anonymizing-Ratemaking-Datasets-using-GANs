# Decoder
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

from gan_scripts.decoder_functions import MultiType, CategoricalActivation

class Decoder(nn.Module):
    """
    Decodes a batch of given data
    
    Init: 
      z_size:    size of the z vector, which is also the size of the encoder output
      output_size: size of the actual data
      hidden_sizes: list of hidden sizes, if empty (default) then there will
                    be one layer with data_size in_features,
                    and z_size out_features
    
    Forward:
      inputs: encoded data
    
    Returns:
      decoded data
    """
    def __init__(self, z_size, output_size, hidden_sizes=[]):
        super(Decoder, self).__init__()

        hidden_activation = nn.Sigmoid()
        previous_layer_size = z_size
        hidden_layers = []

        for layer_size in hidden_sizes:
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size, bias=False))
            hidden_layers.append(hidden_activation)
            previous_layer_size = layer_size

        if len(hidden_layers) > 0:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        else:
            self.hidden_layers = None

        if type(output_size) is list:
            self.output_layer = MultiType(previous_layer_size, output_size)
        else:
            raise Exception("Invalid output size.")

    def forward(self, code, training=False, temperature=None):
        if self.hidden_layers is None:
            hidden = code
        else:
            hidden = self.hidden_layers(code)

        return self.output_layer(hidden, training=training, temperature=temperature)
