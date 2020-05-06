#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:47:45 2019

@author: Elijah
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 20 2019
@author: Elijah Harmon
"""

# Ensure matrix multiplication only uses one core
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

## Import packages
import torch
import pandas as pd
import numpy as np
import statsmodels.api as sm
import csv
from patsy import dmatrices
pd.set_option('display.float_format', lambda x: '%.3f' % x)

## Import created modules
from gan_scripts.undo_dummy import back_from_dummies


# Sets the number of cores used on the server
torch.set_num_threads(1)

# Load and edit data
policy1 = pd.read_csv("./datasets/policy_dat_v2.csv")
policy1['ClaimNb'] = policy1['ClaimNb'].astype('category')
policy1['ExposureCat'] = policy1['Exposure_cat'].astype('category')
policy1['DensityCat'] = policy1['Density_cat'].astype('category')

policy_cat = pd.get_dummies(policy1.loc[:,["ClaimNb",
                                           "Power",
                                           "Brand",
                                           "Gas",
                                           "Region",
                                           "ExposureCat",
                                           "DensityCat"]])
cont_vars = policy1[['CarAge','DriverAge']]

# Center and scale continuous variables
cont_vars2 = (cont_vars - np.mean(cont_vars))/np.std(cont_vars)
pol_dat = pd.concat([cont_vars2.reset_index(drop=True), policy_cat], axis=1) 


# Split the data in to train, test, and validation sets (70,15,15)
np.random.seed(seed=123)
all_inds = np.arange(0,(pol_dat.shape[0]-1))
test_inds = np.random.choice(all_inds, size=np.floor(pol_dat.shape[0]*.15).astype('int'), replace=False, p=None)
second_inds = np.setdiff1d(all_inds, test_inds)
val_inds = np.random.choice(second_inds, size=np.floor(pol_dat.shape[0]*.15).astype('int'), replace=False, p=None)
third_inds = np.setdiff1d(all_inds, np.concatenate((val_inds,test_inds)))
train_inds = np.setdiff1d(all_inds, np.concatenate((val_inds,test_inds)))
fourth_inds = np.setdiff1d(all_inds, val_inds)
  
test = pol_dat.iloc[test_inds]
val = pol_dat.iloc[val_inds]
train = pol_dat.iloc[train_inds]
sampling = pol_dat.iloc[fourth_inds]

  
  # Wrangle sampling data
sd = back_from_dummies(sampling)
sd['ClaimNb'] = sd['ClaimNb'].astype('int')
y_sample, X_sample = dmatrices('ClaimNb ~ CarAge + DriverAge + Power + Brand + Gas + Region + DensityCat',
                           data=sd,
                           return_type='dataframe')
sd['Exposure'] = sd['ExposureCat'].astype('float32')/11

valid = back_from_dummies(val)
valid['ClaimNb'] = valid['ClaimNb'].astype('int')
y_valid, X_valid = dmatrices('ClaimNb ~ CarAge + DriverAge + Power + Brand + Gas + Region + DensityCat',
                           data=valid,
                           return_type='dataframe')
valid['Exposure'] = valid['ExposureCat'].astype('float32')/11

pois_metric = []

for i in range(5000):
    a_inds = np.random.choice(fourth_inds, size=np.floor(sampling.shape[0]*(7/8.5)).astype('int'), replace=False, p=None)
    b_inds = np.setdiff1d(fourth_inds, a_inds)
    
  
    a_1 = pol_dat.iloc[a_inds]
    b_1 = pol_dat.iloc[b_inds]
    
    a_1 = back_from_dummies(a_1)
    a_1['ClaimNb'] = a_1['ClaimNb'].astype('float32')/11
    b_1 = back_from_dummies(b_1)
    b_1['ClaimNb'] = b_1['ClaimNb'].astype('float32')/11
    
    y_a, X_a = dmatrices('ClaimNb ~ CarAge + DriverAge + Power + Brand + Gas + Region + DensityCat',
                           data=a_1,
                           return_type='dataframe')
    y_b, X_b = dmatrices('ClaimNb ~ CarAge + DriverAge + Power + Brand + Gas + Region + DensityCat',
                           data=b_1,
                           return_type='dataframe')
    a_1['Exposure'] = a_1['ExposureCat'].astype('float32')/11
    b_1['Exposure'] = b_1['ExposureCat'].astype('float32')/11
    
  # Fit poisson Models
    poisson_mod_a = sm.GLM(y_a,X_a,family = sm.families.Poisson(), offset = np.log(a_1['Exposure'])).fit()
    original_params = poisson_mod_a.params
    
    poisson_mod_b = sm.GLM(y_b,X_b,family = sm.families.Poisson(), offset = np.log(b_1['Exposure'])).fit()
    original_params = poisson_mod_b.params


    # make predictions on validation data with models trained on a and b
    
    errors_pois = poisson_mod_a.predict(X_valid) - poisson_mod_b.predict(X_valid)
  
    pois_metric.append(round(np.mean(errors_pois), 10))
    


with open('poisson_metric.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(pois_metric)

csvFile.close()
