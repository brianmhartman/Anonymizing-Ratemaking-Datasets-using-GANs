# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 10:07:20 2019

@author: joshj
"""

# import modules
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices
pd.set_option('display.float_format', lambda x: '%.2f' % x)

## Import created modules
from gan_scripts.auto_loader import PolicyDataset
from gan_scripts.autoencoder import AutoEncoder
from gan_scripts.generator2_v2 import Generator2
from gan_scripts.discriminator2_v3 import Discriminator2
from gan_scripts.gradiant_penalty import calculate_gradient_penalty
from gan_scripts.undo_dummy import back_from_dummies

# Load Data

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

cont_vars2 = (cont_vars - np.mean(cont_vars))/np.std(cont_vars)

pol_dat = pd.concat([cont_vars2.reset_index(drop=True), policy_cat], axis=1)

pol_dat  = pol_dat.sample(n = 10000, random_state = 12)

td = back_from_dummies(pol_dat)
td['ClaimNb'] = td['ClaimNb'].astype('int')
y_real, X_real = dmatrices('ClaimNb ~ CarAge + DriverAge + Power + Brand + Gas + Region + DensityCat',
                 data=td,
                 return_type='dataframe')
td['Exposure'] = td['ExposureCat'].astype('float32')/11
def xpxixpy(X,y):
            return np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,y))
xy = xpxixpy(X_real,y_real)
disc_add_rows = xy.shape[0]
poisson_mod = sm.GLM(y_real,X_real,family = sm.families.Poisson(), offset = td['Exposure']).fit()
original_params = poisson_mod.params
lower = poisson_mod.params - 1.96*poisson_mod.bse  
upper = poisson_mod.params + 1.96*poisson_mod.bse 


# parameters
data_size = pol_dat.shape[1] # number of cols in pol_dat 
z_size = 100
batch_size = 250
variable_sizes = [5,12,7,2,10,12,5] # number of levels in each categorical variable
var_locs = [0,1] # location of continous variables
cont_num = len(var_locs)  
temperature = None

# Generator tuning
gen_hidden_sizes = [100,100,100]
gen_bn_decay = .25
gen_l2_regularization = 0.1
gen_learning_rate = 0.001

# Discriminator tuning
disc_hidden_sizes = [55]
disc_bn_decay = 0
critic_bool = False
mini_batch_bool = True
disc_leaky_param = 0.2
disc_l2_regularization = 0.0
disc_learning_rate = 0.001
penalty = 10

autoencoder = AutoEncoder(data_size = data_size-cont_num,
                          z_size = z_size,
                          encoder_hidden_sizes=[],
                          decoder_hidden_sizes=[],
                          variable_sizes= variable_sizes 
                          )
autoencoder.load_state_dict(torch.load('./saved_parameters/autoencoder_190220')) 


auto_data = PolicyDataset(pol_dat, var_locs)
auto_loader = DataLoader(auto_data,
                         batch_size = batch_size,
                         pin_memory = True, 
                         shuffle = True
                         )

generator = Generator2(
    noise_size = z_size,
    output_size =  z_size + cont_num,
    hidden_sizes = gen_hidden_sizes,
    bn_decay = gen_bn_decay
    )

discriminator = Discriminator2(
    input_size = data_size,
    hidden_sizes= disc_hidden_sizes,
    bn_decay = disc_bn_decay,  # no batch normalization for the critic
    critic = critic_bool,
    leaky_param = disc_leaky_param,         # parameter for leakyRelu
    mini_batch = mini_batch_bool,
    add_rows = disc_add_rows
    )

optim_gen = optim.Adam(generator.parameters(),
                       weight_decay= gen_l2_regularization,
                       lr= gen_learning_rate
                       )

optim_disc = optim.Adam(discriminator.parameters(),
                        weight_decay= disc_l2_regularization,
                        lr= disc_learning_rate
                        )    

def generate_data(trained_generator):
    test_noise = Variable(torch.FloatTensor(pol_dat.shape[0], z_size).normal_())
    with torch.no_grad():
        test_code = trained_generator(test_noise, training = False)
    test_cat_partg = test_code[:,cont_num:(z_size+cont_num)]
    test_cont_partg = test_code[:,0:cont_num]
    test_gen_features = torch.cat([test_cont_partg,
                              autoencoder.decode(test_cat_partg,
                                                 training=False,
                                                 temperature=temperature)],1)
    return(test_gen_features)
    

# option to load pretrained parameters. Make sure all of the parameters of the pretrained
# Model match up with the parameters you inilized above 

#generator.load_state_dict(torch.load('./training_iterations/test_start/generator_4_4050_50'))
#optim_gen.load_state_dict(torch.load('./training_iterations/test_start/gen_optim_3_3750_56'))
#discriminator.load_state_dict(torch.load('./training_iterations/test_start/discriminator_3_3850_16'))
#optim_disc.load_state_dict(torch.load('./training_iterations/test_start/disc_optim_3_3850_16'))


epochs = 1000
disc_epochs = 2
gen_epochs = 1
autoencoder.train(mode=True)
generator.train(mode=True)
discriminator.train(mode=True)
disc_losses = []
gen_losses = []
disc_loss = torch.tensor(9999)
gen_loss = torch.tensor(9999)
cont_num = len(var_locs)
loop = tqdm(total = epochs, position = 0, leave = False)
for epoch in range(epochs):
    for d_epoch in range(disc_epochs):  
        for c1,c2 in auto_loader:
            batch = torch.cat([c2,c1],1) 
            optim_disc.zero_grad()
            # train discriminator with real data
            real_features = Variable(batch)
            real_pred = discriminator(real_features)
            # the disc outputs high numbers if it thinks the data is real, we take the negative of this
            real_loss = - real_pred.mean(0).view(1)  
            real_loss.backward()

            # then train the discriminator only with fake data
            cont_num = len(var_locs)
            noise = Variable(torch.FloatTensor(len(batch), z_size).normal_())
            fake_code = generator(noise)
            cat_part = fake_code[:,cont_num:(z_size+cont_num)]
            cont_part = fake_code[:,0:cont_num]
            fake_features = torch.cat([cont_part,
                                       autoencoder.decode(cat_part,
                                       training=True,
                                       temperature=temperature)],1)
            fake_features = fake_features.detach()  # do not propagate to the generator
            fake_pred = discriminator(fake_features)
            fake_loss = fake_pred.mean(0).view(1)
            fake_loss.backward()
            
            # this is the magic from WGAN-GP
            gradient_penalty = calculate_gradient_penalty(discriminator, penalty, real_features, fake_features)
            gradient_penalty.backward()

            # finally update the discriminator weights
            optim_disc.step()

            disc_loss = real_loss + fake_loss + gradient_penalty
            disc_losses.append(disc_loss.item())
            # Delete to prevent memory leakage
            del gradient_penalty
            del fake_loss
            del real_loss
            del disc_loss
    

    for g_epoch in range(gen_epochs):
        optim_gen.zero_grad()

        noise = Variable(torch.FloatTensor(len(batch), z_size).normal_())
        gen_code = generator(noise)
        cat_partg = gen_code[:,cont_num:(z_size+cont_num)]
        cont_partg = gen_code[:,0:cont_num]
        gen_features = torch.cat([cont_partg,
                                  autoencoder.decode(cat_partg,
                                  training=True,
                                  temperature=temperature)],1)
        gen_pred = discriminator(gen_features)

        gen_loss = - gen_pred.mean(0).view(1)
        gen_loss.backward()

        optim_gen.step()

        gen_loss = gen_loss
        gen_losses.append(gen_loss.item())
        del gen_loss 
            
    loop.set_description('epoch:{}, disc_loss:{:.4f}, gen_loss:{:.4f}'.format(epoch, disc_losses[epoch], gen_losses[epoch]))
    loop.update(1) 
    # analyze poisson regression parameters every 20 epochs
    if(epoch % 1 == 0):
        generated_data = generate_data(generator)
        df1 = pd.DataFrame(generated_data.data.numpy())
        df1.columns = list(pol_dat)
        df2 = back_from_dummies(df1)
        df2['ClaimNb'] = df2['ClaimNb'].astype('int')
        y_gen, X_gen = dmatrices('ClaimNb ~ CarAge + DriverAge + Power + Brand + Gas + Region + DensityCat',
                             data=df2,
                             return_type='dataframe')
        df2['Exposure'] = df2['ExposureCat'].astype('float32')/11
        poisson_mod_gen = sm.GLM(y_gen,X_gen,family = sm.families.Poisson(), offset = df2['Exposure']).fit()
        pois_df = pd.concat([original_params.reset_index(drop=True),
                 lower.reset_index(drop=True),
                 upper.reset_index(drop=True),
                 poisson_mod_gen.params.reset_index(drop=True),
                 ((poisson_mod_gen.params > lower) & (poisson_mod_gen.params < upper)).reset_index(drop=True)],
        axis = 1)

        pois_df.columns = ['Real', 'Lower', 'Upper', 'Generated', 'IN 95']
        pois_df.index = poisson_mod.params.index
        print(pois_df)

      
