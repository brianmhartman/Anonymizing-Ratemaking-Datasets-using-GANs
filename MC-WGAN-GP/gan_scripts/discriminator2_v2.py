
from __future__ import print_function

import torch.nn as nn
import torch


class Discriminator2(nn.Module):

    def __init__(self,
                 input_size,                # Number of columns in data
                 hidden_sizes= (256, 128),   # Size of hidden layers for discriminator
                 bn_decay=0.01,             # Batch norm decay parameter
                 critic=False,              # If false then sigmoid output, if true linear ourput
                 leaky_param = 0.2,         # parameter for leakyRelu
                 mini_batch = False         # Is there minibach averaging? 
                 ):
        super(Discriminator2, self).__init__()

        hidden_activation = nn.LeakyReLU(leaky_param)
        
        if mini_batch:
            previous_layer_size = input_size*2
        else:
            previous_layer_size = input_size
        
        layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(previous_layer_size, layer_size))
            if layer_number > 0 and bn_decay > 0:
                layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))
            layers.append(hidden_activation)
            previous_layer_size = layer_size

        layers.append(nn.Linear(previous_layer_size, 1))

        # the critic has a linear output
        if not critic:
            layers.append(nn.Sigmoid())
        self.mini_batch = mini_batch
        self.model = nn.Sequential(*layers)
        
    def minibatch_averaging(self, inputs):
        """
        This method is explained in the MedGAN paper.
        """
        mean_per_feature = torch.mean(inputs, 0)
        mean_per_feature_repeated = mean_per_feature.repeat(len(inputs), 1)
        return torch.cat((inputs, mean_per_feature_repeated), 1)

    
    def forward(self, inputs):
        if self.mini_batch:
            inputs = self.minibatch_averaging(inputs)
        return self.model(inputs).view(-1)

                

        
       
            
    
    
