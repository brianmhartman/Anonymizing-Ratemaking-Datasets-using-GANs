# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 14:25:39 2018

@author: Joshua Meyers
v2 changes switched order of autoloader and disc epochs (lines 87-89)
   deleted losses to hopefully reduce memory leakage
v5 implement xpxixpy in the discriminator to hopefully help multivariate training
   it also now looks at a poisson regression model instead of univariate counts
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
    return [int(i) for i in comma_separated_ints.split(",")]


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

# Scale the continuos variables to help training
cont_vars2 = (cont_vars - np.mean(cont_vars))/np.std(cont_vars)
pol_dat = pd.concat([cont_vars2.reset_index(drop=True), policy_cat], axis=1)

# Take a sample from the original data for faster training
pol_dat = pol_dat.sample(n = 10000, random_state = 12)

# Fit a poisson model for the original data
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
poisson_mod = sm.GLM(y_real,X_real,family = sm.families.Poisson(), offset = np.log(td['Exposure'])).fit()
original_params = poisson_mod.params
lower = poisson_mod.params - 1.96*poisson_mod.bse  
upper = poisson_mod.params + 1.96*poisson_mod.bse 


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
               output_gen_path = './saved_parameters/generator2',
               output_disc_optim_path = './saved_parameters/disc_optim2',
               output_gen_optim_path = './saved_parameters/gen_optim2',
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
    
                # then train the discriminator only with fake data
                cont_num = len(var_locs)
                noise = Variable(torch.FloatTensor(len(batch), z_size).normal_())
                fake_code = generator(noise)
                cat_part = fake_code[:,cont_num:(z_size+cont_num)]
                cont_part = fake_code[:,0:cont_num]
                fake_features = torch.cat([cont_part,
                                           autoencoder.decode(cat_part,
                                           training=False,
                                           temperature=temperature)],1)
                fake_features = fake_features.detach()  # do not propagate to the generator
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

            noise = Variable(torch.FloatTensor(len(batch), z_size).normal_())
            gen_code = generator(noise)
            cat_partg = gen_code[:,cont_num:(z_size+cont_num)]
            cont_partg = gen_code[:,0:cont_num]
            gen_features = torch.cat([cont_partg,
                                      autoencoder.decode(cat_partg,
                                      training=False,
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
        if(epoch % 50 == 0):        
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
            
            generated_data = generate_data(generator)
            df1 = pd.DataFrame(generated_data.data.numpy())
            df1.columns = list(pol_dat)
            df2 = back_from_dummies(df1)
            df2['ClaimNb'] = df2['ClaimNb'].astype('int')
            y_gen, X_gen = dmatrices('ClaimNb ~ CarAge + DriverAge + Power + Brand + Gas + Region + DensityCat',
                             data=df2,
                             return_type='dataframe')
            df2['Exposure'] = df2['ExposureCat'].astype('float32')/11
            poisson_mod_gen = sm.GLM(y_gen,X_gen,family = sm.families.Poisson(), offset = np.log(df2['Exposure'])).fit()
            pois_df = pd.concat([original_params.reset_index(drop=True),
                     lower.reset_index(drop=True),
                     upper.reset_index(drop=True),
                     poisson_mod_gen.params.reset_index(drop=True),
                     ((poisson_mod_gen.params > lower) & (poisson_mod_gen.params < upper)).reset_index(drop=True)],
    axis = 1)

            pois_df.columns = ['Real', 'Lower', 'Upper', 'Generated', 'IN 95']
            pois_df.index = poisson_mod.params.index
            print(pois_df)
            if save_bool:    
                # Option to save parameters to restart training
                torch.save(discriminator.state_dict(), f=output_disc_path+'_'+str(epoch))
                torch.save(generator.state_dict(), f=output_gen_path+'_'+str(epoch))
                torch.save(optim_disc.state_dict(), f=output_disc_optim_path+'_'+str(epoch))
                torch.save(optim_gen.state_dict(), f=output_gen_optim_path+'_'+str(epoch))


def main():
    options_parser = argparse.ArgumentParser(description="Train GAN1.")
    
    options_parser.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="Number of epochs."
    )
    
    options_parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Amount of samples per batch."
    )
    
    options_parser.add_argument(
        "--disc_epochs",
        type=int,
        default=1,
        help="Number of epochs for discriminator."
    )
    
    options_parser.add_argument(
        "--gen_epochs",
        type=int,
        default=1,
        help="Number of epochs for generator."
    )
    
    options_parser.add_argument(
        "--loss_penalty",
        type=float,
        default= 0.1,
        help="Loss penalty for gradiant."
    )
    
    options_parser.add_argument(
        "--z_size",
        type=int,
        default=100,
        help="Size of the random noice vector"
    )

    options_parser.add_argument(
        "--data_size",
        type=int,
        default=158,
        help="Number of Columns in the data"
    )
    
    options_parser.add_argument(
        "--save_bool",
        type=str,
        default= "False",
        help="Should the parameters be saved?"
    )
    
    options_parser.add_argument(
        "--gen_bn_decay",
        type=float,
        default=0.01,
        help="bn_decay for the generator"
    )
    
    options_parser.add_argument(
        "--disc_bn_decay",
        type=float,
        default=0.01,
        help="bn_decay for the discriminator"
    )
     
    options_parser.add_argument(
        "--critic",
        type=str,
        default= "True",
        help="If True linear if False Sigmoid"
    )
    
    options_parser.add_argument(
        "--mini_batch",
        type=str,
        default= "False",
        help="Should minibatch be used?"
    )
    
    options_parser.add_argument(
        "--disc_leaky_param",
        type=float,
        default=0.2,
        help="Parameter for the leaky ReLu activation for discriminator"
    )
    
    options_parser.add_argument(
        "--gen_l2_regularization",
        type=float,
        default=0.001,
        help="L2 regularization for generator."
    )
    
    options_parser.add_argument(
        "--disc_l2_regularization",
        type=float,
        default=0.001,
        help="L2 regularization for discriminator."
    )

    options_parser.add_argument(
        "--disc_learning_rate",
        type=float,
        default=0.001,
        help="Adam learning rate for disciminator."
    )
    
    options_parser.add_argument(
        "--gen_learning_rate",
        type=float,
        default=0.001,
        help="Adam learning rate for generator."
    )
    
    options_parser.add_argument(
        "--var_locs",
        type=str,
        default="0,1,2,3",
        help="location of continous variables"
    )
    
    options_parser.add_argument(
        "--variable_sizes",
        type=str,
        default="5,12,7,2,10",
        help="size of each of the categorical variables"
    )
    
    options_parser.add_argument(
        "--auto_encoder_loc",
        type=str,
        default='./saved_parameters/autoencoder',
        help="location of the trained autoencoder"
    )
    
    options_parser.add_argument(
        "--generator_hidden_sizes",
        type=str,
        default="100,100,100",
        help="Hidden sizes for the generator"
    )
    
    options_parser.add_argument(
        "--discriminator_hidden_sizes",
        type=str,
        default="100,100,100",
        help="Hidden sizes for the discriminator"
    )
    
    options_parser.add_argument(
        "--disc_save_loc",
        type=str,
        default='./saved_parameters/discriminator99',
        help="Location to save the discriminator parameters"
    )
    
    options_parser.add_argument(
        "--gen_save_loc",
        type=str,
        default='./saved_parameters/generator99',
        help="Location to save the generator parameters"
    )
    
    options_parser.add_argument(
        "--disc_optim_save_loc",
        type=str,
        default='./saved_parameters/disc_optim99',
        help="Location to save the discriminator optim parameters"
    )
    options_parser.add_argument(
        "--gen_optim_save_loc",
        type=str,
        default= "./saved_parameters/gen_optim99",
        help="Location to save the generator optim parameters"
    )
    options_parser.add_argument(
        "--fig_save_path",
        type=str,
        default= "./saved_parameters/fig99",
        help="Location to save updated figures"
    )
    
    options = options_parser.parse_args()
    cont_num = len(parse_int_list(options.var_locs))
    # Initialize autoencoder
    autoencoder = AutoEncoder(data_size = options.data_size-cont_num,
                          z_size = options.z_size,
                          encoder_hidden_sizes=[],
                          decoder_hidden_sizes=[],
                          variable_sizes= parse_int_list(options.variable_sizes) 
                          )
    autoencoder.load_state_dict(torch.load(options.auto_encoder_loc)) 

    auto_data=PolicyDataset(pol_dat, parse_int_list(options.var_locs))
    auto_loader = DataLoader(auto_data,
                             batch_size = options.batch_size,
                             pin_memory = True, 
                             shuffle = True
                             )
    # Initialize Generator and Discriminator
    generator = Generator2(
        noise_size = options.z_size,
        output_size =  options.z_size+len(parse_int_list(options.var_locs)),
        hidden_sizes = parse_int_list(options.generator_hidden_sizes),
        bn_decay = options.gen_bn_decay
        )
    
    critic_bool = options.critic == 'True'
    mini_batch_bool = options.mini_batch == 'True'
    save_bool = options.save_bool == 'True'
    
    discriminator = Discriminator2(
        input_size = options.data_size,
        hidden_sizes= tuple(parse_int_list(options.discriminator_hidden_sizes)),
        bn_decay = options.disc_bn_decay,  # no batch normalization for the critic
        critic = critic_bool,
        leaky_param = options.disc_leaky_param,         # parameter for leakyRelu
        mini_batch = mini_batch_bool,
        add_rows = disc_add_rows
        )
    
    optim_gen = optim.Adam(generator.parameters(),
                           weight_decay= options.gen_l2_regularization,
                           lr= options.gen_learning_rate
                           )
    
    optim_disc = optim.Adam(discriminator.parameters(),
                            weight_decay= options.disc_l2_regularization,
                            lr= options.disc_learning_rate
                            )    
    train_GAN2(generator,
           discriminator,
           optim_gen,
           optim_disc,
           auto_loader,
           autoencoder = autoencoder,
           z_size = options.z_size,
           epochs = options.num_epochs,
           disc_epochs = options.disc_epochs,
           gen_epochs = options.gen_epochs,
           penalty= options.loss_penalty,
           temperature=None,
           save_bool = save_bool,
           var_locs = parse_int_list(options.var_locs),
           output_disc_path = options.disc_save_loc,
           output_gen_path = options.gen_save_loc,
           output_disc_optim_path = options.disc_optim_save_loc,
           output_gen_optim_path = options.gen_optim_save_loc,
           output_fig_save_path = options.fig_save_path
           ) 
if __name__ == "__main__":
    main()