# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 14:25:39 2018

@author: Joshua Meyers
"""

# Ensure matrix multiplication only uses one core
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

## Import packages
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.ensemble import RandomForestRegressor
pd.set_option('display.float_format', lambda x: '%.3f' % x)

## Import created modules
from gan_scripts.auto_loader import PolicyDataset
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

# Center and scale continuous variables
cont_vars2 = (cont_vars - np.mean(cont_vars))/np.std(cont_vars)
pol_dat = pd.concat([cont_vars2.reset_index(drop=True), policy_cat], axis=1) 

# Take a sampel of the data for quicker training
#pol_dat  = pol_dat.sample(n = 10000, random_state = 12)

# Split the data in to train, test, and validation sets (80,10,10)
np.random.seed(seed=123)
all_inds = np.arange(0,(pol_dat.shape[0]-1))
test_inds = np.random.choice(all_inds, size=np.floor(pol_dat.shape[0]*.15).astype('int'), replace=False, p=None)
second_inds = np.setdiff1d(all_inds, test_inds)
val_inds = np.random.choice(second_inds, size=np.floor(pol_dat.shape[0]*.15).astype('int'), replace=False, p=None)
third_inds = np.setdiff1d(all_inds, np.concatenate((val_inds,test_inds)))
#best_test_inds = np.random.choice(third_inds, size=np.floor(pol_dat.shape[0]*.15).astype('int'), replace=False, p=None)
train_inds = np.setdiff1d(all_inds, np.concatenate((val_inds,test_inds)))

test = pol_dat.iloc[test_inds]
val = pol_dat.iloc[val_inds]
best_test = pol_dat.iloc[val_inds] 
train = pol_dat.iloc[train_inds]


pol_dat = train

# Wrangle train data
td = back_from_dummies(train)
td['ClaimNb'] = td['ClaimNb'].astype('int')
y_real, X_real = dmatrices('ClaimNb ~ CarAge + DriverAge + Power + Brand + Gas + Region + DensityCat',
                 data=td,
                 return_type='dataframe')
td['Exposure'] = td['ExposureCat'].astype('float32')/11
def xpxixpy(X,y):
            return np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,y))
xy = xpxixpy(X_real,y_real)
disc_add_rows = xy.shape[0]


# Fit a poisson Model
poisson_mod = sm.GLM(y_real,X_real,family = sm.families.Poisson(), offset = np.log(td['Exposure'])).fit()
original_params = poisson_mod.params

lower = poisson_mod.params - 1.96*poisson_mod.bse  
upper = poisson_mod.params + 1.96*poisson_mod.bse 


# Fit a random forest
# real_features= X_real
# real_feature_list = list(real_features.columns)
# real_features = np.array(real_features)
# y_rep = np.squeeze(y_real)/np.squeeze(td['Exposure'])
# rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# rf.fit(real_features, y_rep)

# Wrangle Test Data
test2 = back_from_dummies(test)
test2['ClaimNb'] = test2['ClaimNb'].astype('int')
y_test, X_test = dmatrices('ClaimNb ~ CarAge + DriverAge + Power + Brand + Gas + Region + DensityCat',
                 data=test2,
                 return_type='dataframe')
test2['Exposure'] = test2['ExposureCat'].astype('float32')/11
y_test_resp = np.squeeze(y_test)/np.squeeze(test2['Exposure'])


# make predictions on test data with models trained on train data
real_pois_preds = poisson_mod.predict(X_test)
# real_predictions = rf.predict(X_test)
# importances_real = rf.feature_importances_ 


def train_GAN2(generator,
               discriminator,
               optim_gen,
               optim_disc,
               auto_loader,
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
               output_fig_save_path = './saved_parameters/fig1',
               output_data_save_path = './saved_parameters/data_generator2'
               ):
    generator.train(mode=True)
    discriminator.train(mode=True)
    disc_losses = []
    gen_losses = []
    pois_metric = []
    # rf_metric = []
    # rf_imp_metric = []
    disc_loss = torch.tensor(9999)
    gen_loss = torch.tensor(9999)
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
                # Because we are minimizing loss
                real_loss = - real_pred.mean(0).view(1)  
                real_loss.backward()
    
                # then train the discriminator only with fake data
                noise = Variable(torch.FloatTensor(len(batch), z_size).normal_())
                fake_features = generator(noise)
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
                del real_features
                del real_pred
                del noise
                del fake_features
                del fake_pred
                
        
    
        for g_epoch in range(gen_epochs):
            optim_gen.zero_grad()
    
            noise = Variable(torch.FloatTensor(len(batch), z_size).normal_())
            gen_features = generator(noise, training = TRUE)
            gen_pred = discriminator(gen_features)
    
            gen_loss = - gen_pred.mean(0).view(1)
            gen_loss.backward()
    
            optim_gen.step()
    
            gen_loss = gen_loss
            gen_losses.append(gen_loss.item())
            del gen_loss
            del noise
            del gen_features
            del gen_pred
                
        loop.set_description('epoch:{}, disc_loss:{:.4f}, gen_loss:{:.4f}'.format(epoch, disc_losses[epoch], gen_losses[epoch]))
        loop.update(1) 
        # analyze poisson regression parameters every 20 epochs
        if(epoch % 25 == 0):
            with torch.no_grad():
                generated_data = generator(Variable(torch.FloatTensor(best_test.shape[0], z_size).normal_()), training = False)
            df1 = pd.DataFrame(generated_data.data.numpy())
            df1.columns = list(pol_dat)
            df2 = back_from_dummies(df1)
            df2['ClaimNb'] = df2['ClaimNb'].astype('int')
            y_gen, X_gen = dmatrices('ClaimNb ~ CarAge + DriverAge + Power + Brand + Gas + Region + DensityCat',
                                 data=df2,
                                 return_type='dataframe')
            df2['Exposure'] = df2['ExposureCat'].astype('float32')/11
            
            #df2.to_csv(output_data_save_path)
            # Fit poisson Model
            poisson_mod_gen = sm.GLM(y_gen,X_gen,family = sm.families.Poisson(), offset = np.log(df2['Exposure'])).fit()
            
            # Fit Random Forest
            # gen_features= X_gen
            # gen_features = np.array(gen_features)
            # y_gen2 = np.squeeze(y_gen)/np.squeeze(df2['Exposure'])
            # rf_gen = RandomForestRegressor(n_estimators = 1000, random_state = 42)
            # rf_gen.fit(gen_features, np.squeeze(y_gen2))
            
            # gen_predictions = rf_gen.predict(X_test)
            
            # importances_gen = rf_gen.feature_importances_
            
            # Calculate Errors
            errors_pois = poisson_mod_gen.predict(X_test) - real_pois_preds
            # errors_rf = abs(gen_predictions - real_predictions)
            # errors_imp = abs(importances_gen - importances_real)
            
            pois_metric.append(round(np.mean(errors_pois), 4))
            # rf_metric.append(round(np.mean(errors_rf), 4))
            # rf_imp_metric.append(np.mean(errors_imp))
            
            if(epoch > 3) :
                plt.subplot(311)
                plt.plot(pois_metric, label = 'train')
                plt.ylabel('poission Dif')
                
                # plt.subplot(312)
                # plt.plot(rf_metric, label = 'train')
                # plt.ylabel('rf pred dif')
                
                # plt.subplot(313)
                # plt.plot(rf_imp_metric, label = 'train')
                # plt.ylabel('rf imp dif')
                if save_bool:
                    plt.savefig(output_fig_save_path, bbox_inches='tight')
                plt.show()
                plt.clf()
            
            #print('Mean Absolute Difference RF:', round(np.mean(errors_rf), 4))
            print('Mean Absolute Difference Pois:', round(np.mean(errors_pois), 4))
            #print('Mean Absolute Difference RF Imp:', round(np.mean(errors_imp), 4))
            
            del errors_pois
            #del errors_rf
            #del errors_imp
            #del rf_gen
            #del y_gen2
            del poisson_mod_gen
            del y_gen
            del X_gen
            #del importances_gen
            #del gen_predictions 
            del generated_data
            del df1
            del df2
            #del gen_features

            if save_bool:    
                # Option to save parameters to restart training
                torch.save(discriminator.state_dict(), f=output_disc_path)
                torch.save(generator.state_dict(), f=output_gen_path)
                torch.save(optim_disc.state_dict(), f=output_disc_optim_path)
                torch.save(optim_gen.state_dict(), f=output_gen_optim_path)
            

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
        help="Size of the random noise vector"
    )

    options_parser.add_argument(
        "--data_size",
        type=int,
        default=158,
        help="Number of Columns in the data"
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
        "--save_bool",
        type=str,
        default= "False",
        help="Should the parameters be saved?"
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

    auto_data=PolicyDataset(pol_dat, parse_int_list(options.var_locs))
    auto_loader = DataLoader(auto_data,
                             batch_size = options.batch_size,
                             pin_memory = True, 
                             shuffle = True
                             )
    # Initialize Generator and Discriminator
    generator = Generator2(
        noise_size = options.z_size,
        output_size =  [1]*len(parse_int_list(options.var_locs)) +  parse_int_list(options.variable_sizes),
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
           z_size = options.z_size,
           epochs = options.num_epochs,
           disc_epochs = options.disc_epochs,
           gen_epochs = options.gen_epochs,
           penalty= options.loss_penalty,
           temperature=None,
           var_locs = parse_int_list(options.var_locs),
           save_bool = save_bool,
           output_disc_path = options.disc_save_loc,
           output_gen_path = options.gen_save_loc,
           output_disc_optim_path = options.disc_optim_save_loc,
           output_gen_optim_path = options.gen_optim_save_loc,
           output_fig_save_path = options.fig_save_path,
           output_data_save_path = options.gen_save_loc + '_data'
           ) 
if __name__ == "__main__":
    main()
