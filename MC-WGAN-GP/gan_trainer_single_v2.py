#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:33:07 2019

@author: Elijah
"""

## Import packages


# Ensure matrix multiplication only uses one core
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices

## Import created modules
from gan_scripts.auto_loader import PolicyDataset
from gan_scripts.autoencoder import AutoEncoder
from gan_scripts.generator2_v2 import Generator2
from gan_scripts.discriminator2_v3 import Discriminator2
from gan_scripts.gradiant_penalty import calculate_gradient_penalty
from gan_scripts.undo_dummy import back_from_dummies


# Sets the number of cores used on the server
torch.set_num_threads(1)

def parse_int_list(comma_separated_ints):
    if comma_separated_ints is None or comma_separated_ints == "":
        return []
    else: return [int(i) for i in comma_separated_ints.split(",")]

#Load and edit the data
#To speed up training, subset to 3 variables until we get a GAN we like
# Get dummy variables for categorical variables and separate categorical and continuous variables
policy1 = pd.read_csv("./datasets/policy_dat_v2.csv")
policy1 = policy1[['ClaimNb','DriverAge', 'Density_cat']]
policy1['ClaimNb'] = policy1['ClaimNb'].astype('category')
policy1['DensityCat'] = policy1['Density_cat'].astype('category')
policy_cat = pd.get_dummies(policy1.loc[:,["ClaimNb",
                                           "DensityCat"]])
cont_vars = policy1[['DriverAge']]

# Scale the continuous variables to help training and put continuous and categorical variables back together
cont_vars2 = (cont_vars - np.mean(cont_vars))/np.std(cont_vars)
pol_dat = pd.concat([cont_vars2.reset_index(drop=True), policy_cat], axis=1)

# Take a sample from the original data for faster training
pol_dat = pol_dat.sample(n = 10000, random_state = 12)

# Fit a poisson model for the original data
td = back_from_dummies(pol_dat)
td['ClaimNb'] = td['ClaimNb'].astype('int')
y_real, X_real = dmatrices('ClaimNb ~ DensityCat + DriverAge',
                           data=td,
                           return_type='dataframe')
# Try without the xpxixpy
#def xpxixpy(X,y):
#    return np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,y))
#xy = xpxixpy(X_real, y_real)
#disc_add_rows = xy.shape[0]
poisson_mod = sm.GLM(y_real,X_real, family = sm.families.Poisson()).fit()
original_params = poisson_mod.params
lower = poisson_mod.conf_int()[0]
upper = poisson_mod.conf_int()[1]

def train_GAN2(generator,
               discriminator,
               optim_gen,
               optim_disc,
               auto_loader,
               autoencoder,
               z_size,
               epochs = 500,
               disc_epochs = 2,
               gen_epochs = 3,
               penalty=0.1,
               temperature=None,
               var_locs = [0,1,2,3],
               save_bool = False,
               output_disc_path = './saved_parameters/discriminator2',
               output_gen_path = './saved_paramteters/generator2',
               output_disc_optim_path = './saved_parameters/disc_optim2',
               output_fig_save_path = './saved_parameters/fig1'
               ):
    autoencoder.train(mode=False)
    generator.train(mode=True)
    discriminator.train(mode=True)
    disc_losses = []
    gen_losses = []
    disc_loss = torch.tensor(9999)
    gen_loss = torch.tensor(9999)
    cont_num = len(var_locs)
    loop = tqdm(total = epochs, position = 0, leave = False)
    for epoch in range(epochs):
        #loop = tqdm(total = len(auto_loader), position = 0, leave = False)
        for d_epoch in range(disc_epochs):
            for c1,c2 in auto_loader:
                batch = torch.cat([c2,c1],1)
                optim_disc.zero_grad()
                # train discriminator with real data
                real_features = Variable(batch)
                real_pred = discriminator(real_features)
                
                # the disc output high numbers if it thinks the data is real, we take the negative of this
                real_loss = - real_pred.mean(0).view(1)
                real_loss.backward()
                
                #Then train the discriminator only with fake data
                cont_num = len(var_locs)
                noise = Variable(torch.FloatTensor(len(batch), z_size).normal())
                fake_code = generator(noise)
                cat_part = fake_code[:,cont_num:(z_size+cont_num)]
                cont_part = fake_code[:,0:cont_num]
                fake_features = torch.cat([cont_part,
                                           autoencoder.decode(cat_part,
                                           training=False,
                                           temperature=temperature)],1)
                fake_features = fake_features.detach() # do not propogate to the generator
                fake_pred = discriminator(fake_features)
                fake_loss = fake_pred.mean(0).view(1)
                fake_loss.backward()
                
                # this is the magic from WGAN-GP
                gradient_penalty = calculate_gradient_penalty(discriminator, penalty, real_features, fake_features)
                gradient_penalty.backward()
                
                # finally update the discriminator weights
                # using two separated batches is another trick to improve GAN training
                optim_disc.step()
                
                disc_loss = real_loss + fake_loss + gradient_penalty
                disc_losses.append(disc_loss.item())
                del gradient_penalty
                del fake_loss
                del real_loss
                del disc_loss
            
            
            for g_epoch in range(gen_epochs):
                optim_gen.zero_grad()
                
                noise = Variable(torch.FloatTensor(len(batch), z_size).normal())
                gen_code = generator(noise)
                cat_partg = gen_code[:,cont_num:(z_size+cont_num)]
                cont_partg = gen_code[:,0:cont_num]
                gen_features = torch.cat([cont_partg,
                                          autoencoder.decode(cat_partg,
                                          training=False,
                                          temperature=temperature)],1)
                gen_pred = disciminator(gen_features)
                gen_loss = - gen_pred.mean(0).view(1)
                gen_loss.backward()
                
                optim_gen.step()
                
                gen_loss = gen_loss
                gen_losses.append(gen_loss.item())
                del gen_loss
                loop.set_description('epoch:{}, disc_loss:{:.4f}, gen_loss:{:.4f}'.format(epoch, disc_losses[epoch], gen_losses[epoch]))
                loop.update(1)
                
    



            














