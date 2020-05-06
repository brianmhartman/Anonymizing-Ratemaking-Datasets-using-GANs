# -*- coding: utf-8 -*-
"""
This scipt allows you to walk through the generator. Start with tester_multicategorical 
or tester_singleoutput first
"""


from __future__ import print_function
import torch.nn as nn
from gan_scripts.singleoutput import SingleOutput
from gan_scripts.multioutput import MultiCategorical


training = True

hidden_activation = nn.ReLU()

previous_layer_size = noise_size
hidden_sizes = gen_hidden_sizes
bn_decay = gen_bn_decay
hidden_layers = []
noise = Variable(torch.FloatTensor(100, z_size).normal_())


for layer_number, layer_size in enumerate(hidden_sizes):
    hidden_layers.append(nn.Linear(previous_layer_size, layer_size))
    if layer_number > 0 and bn_decay > 0:
        hidden_layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))
    hidden_layers.append(hidden_activation)
    previous_layer_size = layer_size

if len(hidden_layers) > 0:
    hidden_layers = nn.Sequential(*hidden_layers)
else:
    hidden_layers = None

# Define size of final output layer
if type(output_size) is int:
    output = SingleOutput(previous_layer_size, output_size)
elif type(output_size) is list:
    output = MultiCategorical(previous_layer_size, output_size)
else:
    raise Exception("Invalid output size.")


def forward(noise, training=False, temperature=None):
    if hidden_layers is None:
        hidden = noise
    else:
        hidden = hidden_layers(noise)
    
    return output(hidden, training=training, temperature=temperature)


output(hidden, training=False, temperature=temperature)
output(hidden, training=False, temperature=temperature).shape

output_size = [1,1,5,12,7,2,10,12,5]
output_size = 102
output = MultiCategorical(previous_layer_size, output_size)