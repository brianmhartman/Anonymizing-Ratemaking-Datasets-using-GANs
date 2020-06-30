# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 10:19:04 2018
This script trains the autoencoder

@author: Josh Meyers
"""




## Import packages
import pandas as pd
import numpy as np
import torch

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


## Import created modules
from gan_scripts.auto_loader import PolicyDataset
from gan_scripts.autoencoder import AutoEncoder, autoencoder_loss

##
# Load and edit data
policy1=pd.read_csv("./datasets/policy_dat_v2.csv")
policy1['ClaimNb'] = policy1['ClaimNb'].astype('category')
policy1['Exposure_cat'] = policy1['Exposure_cat'].astype('category')
policy1['Density_cat'] = policy1['Density_cat'].astype('category')

policy_cat = pd.get_dummies(policy1.loc[:,["ClaimNb",
                                           "Power",
                                           "Brand",
                                           "Gas",
                                           "Region",
                                           "Exposure_cat",
                                           "Density_cat"]])
cont_vars = policy1[['CarAge','DriverAge']]
pol_dat = pd.concat([cont_vars.reset_index(drop=True), policy_cat], axis=1)


# parameters
data_size = 55
z_size = 55
batch_size = 10000
variable_sizes = [5,12,7,2,10,12,5] # number of levels in each categorical variable
var_locs = [0,1] # location of continous variables
temperature=None

# Data
auto_data=PolicyDataset(pol_dat, var_locs, small_test = None)
auto_loader = DataLoader(auto_data,
                          batch_size = batch_size,
                          pin_memory = True, 
                          shuffle = True)

autoencoder = AutoEncoder(data_size = data_size-len(var_locs),
                          z_size = z_size,
                          encoder_hidden_sizes=[],
                          decoder_hidden_sizes=[],
                          variable_sizes= variable_sizes )



def train_autoencoder(autoencoder,
                      auto_loader,
                      n_epoch = 4,
                      l2_regularization=0.001,
                      learning_rate= 0.003,
                      data_size = data_size,
                      z_size = z_size,
                      variable_sizes = variable_sizes ,# number of levels in each categorical variable
                      var_locs = [0,1],
                      output_optim_path = './saved_parameters/autoencoder_optim',
                      output_autoencoder_path = './saved_parameters/autoencoder_190227'
                      ):
    
    optimizer = optim.Adam(autoencoder.parameters(), weight_decay=l2_regularization, lr=learning_rate)
    training = optimizer is not None
    temperature=None
    losses = []
    for epoch in range(n_epoch):
      epoch_loss = []
      loop = tqdm(total = len(auto_loader), position = 0, leave = False)
      for data, cont_var in auto_loader:
        optimizer.zero_grad()
        #data = data.to(device) 
        _, batch_reconstructed = autoencoder(data, training=training, temperature=temperature)
        loss = autoencoder_loss(reconstructed = batch_reconstructed,
                                original = data,
                                variable_sizes = variable_sizes)
        loss.backward()
        optimizer.step()
        loop.set_description('epoch:{}, loss:{:.4f}'.format(epoch, loss))
        loop.update(1)
        epoch_loss.append(loss.item())
      losses.append(np.mean(epoch_loss))
      torch.save(autoencoder.state_dict(),f=output_autoencoder_path)
    torch.save(optimizer.state_dict(),f=output_optim_path)
    print('DONE')
    
train_autoencoder(autoencoder,
                  auto_loader,
                  data_size = data_size,
                  n_epoch = 5,
                  l2_regularization=0.00001,
                  learning_rate= 0.005,
                  z_size = z_size,
                  variable_sizes = variable_sizes,# number of levels in each categorical variable
                  var_locs = var_locs,
                  output_optim_path = './saved_parameters/autoencoder_optim_190228',
                  output_autoencoder_path = './saved_parameters/autoencoder_190228') 

autoencoder = AutoEncoder(data_size = data_size-len(var_locs),
                          z_size = z_size,
                          encoder_hidden_sizes=[],
                          decoder_hidden_sizes=[],
                          variable_sizes= variable_sizes )  
autoencoder.load_state_dict(torch.load('./saved_parameters/autoencoder_190220'))               


for j,k in auto_loader:
    data = j
_, batch_reconstructed = autoencoder(data, training=True, temperature=temperature)


print(batch_reconstructed[2])
data[2,0:35]

np.round((batch_reconstructed[2] - data[2]).data.numpy(),2)

sum((batch_reconstructed - data).data.numpy() > .)

data.shape

data.data.numpy()


from scipy import stats
pd.DataFrame(data.data.numpy()[:,0]).iloc[:, 0].value_counts()

plt.hist(data.data.numpy()[:,2])

freq(data.data.numpy()[:,2])

data.data.numpy()[:,0].shape

data.data.numpy()[:,0].astype('category')

for j,k in auto_loader:
    print(j[1,0:35])