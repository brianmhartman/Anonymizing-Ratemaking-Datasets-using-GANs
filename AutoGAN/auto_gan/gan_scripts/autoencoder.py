from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from gan_scripts.encoder import Encoder
from gan_scripts.decoder_functions import MultiType, CategoricalActivation
from gan_scripts.decoder import Decoder

class AutoEncoder(nn.Module):

    def __init__(self, data_size =31, z_size=128, encoder_hidden_sizes=[], decoder_hidden_sizes=[],
                 variable_sizes=None):

        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(data_size = data_size,
                               z_size = z_size,
                               hidden_sizes=encoder_hidden_sizes)

        self.decoder = Decoder(z_size = z_size,
                               output_size = variable_sizes,
                               hidden_sizes= decoder_hidden_sizes)

    def forward(self, inputs, training=True, temperature=None):
        code = self.encode(inputs)
        reconstructed = self.decode(code, training=training, temperature=temperature)
        return code, reconstructed

    def encode(self, inputs):
        code = self.encoder(inputs)
        return code

    def decode(self, code, training=True, temperature=None):
        return self.decoder(code, training=training, temperature=temperature)
      
def autoencoder_loss(reconstructed, original, variable_sizes):
    loss = 0
    start = 0
    for variable_size in variable_sizes:
      end = start + variable_size
      batch_reconstructed_variable = reconstructed[:, start:end]
      batch_target = torch.argmax(original[:, start:end], dim=1)
      loss += F.cross_entropy(batch_reconstructed_variable, batch_target)
      start = end
    return loss