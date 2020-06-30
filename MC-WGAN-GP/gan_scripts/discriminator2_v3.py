"""
This script is very similar to the scrip at
https://github.com/rcamino/multi-categorical-gans/tree/master/multi_categorical_gans/methods/general
I changed it so that you can choose the options you want, e.g. critic or not critic

Also the minibatch_xxxy is completly new
"""

from __future__ import print_function

import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from gan_scripts.undo_dummy import back_from_dummies
from patsy import dmatrices

class Discriminator2(nn.Module):

    def __init__(self,
                 input_size,                # Number of columns in data
                 hidden_sizes= (256, 128),   # Size of hidden layers for discriminator
                 bn_decay=0.01,             # Batch norm decay parameter
                 critic=False,              # If false then sigmoid output, if true linear ourput
                 leaky_param = 0.2,         # parameter for leakyRelu
                 mini_batch = True,
                 add_rows = 56 # Is there minibach averaging? 
                 ):
        super(Discriminator2, self).__init__()

        hidden_activation = nn.LeakyReLU(leaky_param)
        
        if mini_batch:
            previous_layer_size = input_size + add_rows
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
        self.add_rows = add_rows
        
    def minibatch_xxxy(self,inputs):    
        def xpxixpy(X,y,size):
            xpp = np.dot(np.linalg.inv(np.dot(X.T,X)+ np.diag(np.repeat(.0001,X.shape[1]))), np.dot(X.T,y))
            xpp = np.append(np.squeeze(xpp).astype(dtype = 'float32'),np.random.normal(0,.0000001,size - len(xpp)))
            return torch.FloatTensor(xpp)
            
        df1 = pd.DataFrame(inputs.data.numpy())
        df1.columns = ['CarAge', 'DriverAge', 'ClaimNb_0', 'ClaimNb_1', 'ClaimNb_2', 'ClaimNb_3', 'ClaimNb_4',
                        'Power_d', 'Power_e', 'Power_f', 'Power_g', 'Power_h', 'Power_i', 'Power_j', 'Power_k',
                         'Power_l', 'Power_m', 'Power_n', 'Power_o', 'Brand_Fiat', 'Brand_Japanese (except Nissan) or Korean',
                          'Brand_Mercedes, Chrysler or BMW', 'Brand_Opel, General Motors or Ford', 'Brand_Renault, Nissan or Citroen',
                           'Brand_Volkswagen, Audi, Skoda or Seat', 'Brand_other', 'Gas_Diesel', 'Gas_Regular', 'Region_R11',
                            'Region_R23', 'Region_R24', 'Region_R25', 'Region_R31', 'Region_R52', 'Region_R53', 'Region_R54', 'Region_R72',
                             'Region_R74', 'ExposureCat_1', 'ExposureCat_2', 'ExposureCat_3', 'ExposureCat_4', 'ExposureCat_5', 'ExposureCat_6',
                             'ExposureCat_7', 'ExposureCat_8', 'ExposureCat_9', 'ExposureCat_10', 'ExposureCat_11', 'ExposureCat_12', 'DensityCat_1',
                              'DensityCat_2', 'DensityCat_3', 'DensityCat_4', 'DensityCat_5']
        df2 = back_from_dummies(df1)
        df2['ClaimNb'] = df2['ClaimNb'].astype('int')
        y, X = dmatrices('ClaimNb ~ CarAge + DriverAge + Power + Brand + Gas + Region + DensityCat',
                         data=df2,
                         return_type='dataframe')
        if (np.sum(y) == 0).bool():
            y[:1] = 1
        return torch.cat((inputs, xpxixpy(X,y,self.add_rows).repeat(len(inputs),1)), 1)
    
    def minibatch_averaging(self, inputs):
        """
        This method is explained in the MedGAN paper.
        """
        mean_per_feature = torch.mean(inputs, 0)
        mean_per_feature_repeated = mean_per_feature.repeat(len(inputs), 1)
        return torch.cat((inputs, mean_per_feature_repeated), 1)

    
    def forward(self, inputs):
        if self.mini_batch:
           # inputs = self.minibatch_xxxy(inputs)
            inputs = self.minbatch_averaging(inputs)
        return self.model(inputs).view(-1)

                        
       
            
    
    
