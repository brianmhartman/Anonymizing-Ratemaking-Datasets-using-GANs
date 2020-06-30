# -*- coding: utf-8 -*-
"""
This script runs the multi GAN and allows you to step through each part

# divide y by exposure in xpxixpy
"""

# import modules
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
from sklearn.model_selection import train_test_split
pd.set_option('display.float_format', lambda x: '%.2f' % x)

## Import created modules
from gan_scripts.auto_loader import PolicyDataset
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

# Center and scale continuous variables
cont_vars2 = (cont_vars - np.mean(cont_vars))/np.std(cont_vars)
pol_dat = pd.concat([cont_vars2.reset_index(drop=True), policy_cat], axis=1)

# Take a sampel of the data for quickly training
pol_dat  = pol_dat.sample(n = 10000, random_state = 12)


# Split the data in to train, test, and validation sets (80,10,10)
all_inds = np.arange(0,(pol_dat.shape[0]-1))
test_inds = np.random.choice(all_inds, size=np.floor(pol_dat.shape[0]*.1).astype('int'), replace=False, p=None)
second_inds = np.setdiff1d(all_inds, test_inds)
val_inds = np.random.choice(second_inds, size=np.floor(pol_dat.shape[0]*.1).astype('int'), replace=False, p=None)
train_inds = np.setdiff1d(second_inds, val_inds)
train_2 = np.random.choice(train_inds, size=np.floor(pol_dat.shape[0]*.4).astype('int'), replace=False, p=None)

test = pol_dat.iloc[test_inds]
val = pol_dat.iloc[val_inds]
train = pol_dat.iloc[train_inds]
train_half = pol_dat.iloc[train_2]

pol_dat = train

# Wrangle train data
td = back_from_dummies(train_half)
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
poisson_mod = sm.GLM(y_real,X_real,family = sm.families.Poisson(), offset = td['Exposure']).fit()
original_params = poisson_mod.params

lower = poisson_mod.params - 1.96*poisson_mod.bse  
upper = poisson_mod.params + 1.96*poisson_mod.bse 


# Fit a random forest
real_features= X_real
real_feature_list = list(real_features.columns)
real_features = np.array(real_features)
y_rep = np.squeeze(y_real)/np.squeeze(td['Exposure'])
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(real_features, y_rep)

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
real_predictions = rf.predict(X_test)
importances_real = rf.feature_importances_

"""
This next section contains everything that we can tune in the GAN
"""

# Information about the size of the data
data_size = pol_dat.shape[1] # number of cols in pol_dat 
var_locs = [0,1] # tells us where the continous variables are 


# parameters
z_size = 100 # how big is the random vector fed into the generator
# we should only need the 55?
batch_size = 1500
temperature = None # comes into play with the categorical activation see multioutput.py

# Generator tuning
gen_hidden_sizes = [100,100,100]
gen_bn_decay = .25
gen_l2_regularization = 0.1
gen_learning_rate = 0.001
noise_size = z_size
output_size = [1,1,5,12,7,2,10,12,5]  # how many categories with in each variable


# Discriminator tuning
disc_hidden_sizes = [55,55]
disc_bn_decay = .2
critic_bool = True # if false then between 0 and 1
mini_batch_bool = True
disc_leaky_param = 0.2
disc_l2_regularization = 0.0
disc_learning_rate = 0.001
penalty = 1 ## deals with gradiant penalty

auto_data = PolicyDataset(pol_dat, var_locs)
auto_loader = DataLoader(auto_data,
                         batch_size = batch_size,
                         pin_memory = True, 
                         shuffle = True
                         )

# initilize generator and discriminator
generator = Generator2(
    noise_size = noise_size,
    output_size =  output_size,
    hidden_sizes = gen_hidden_sizes,
    bn_decay = gen_bn_decay
    )

discriminator = Discriminator2(
    input_size = data_size,
    hidden_sizes= disc_hidden_sizes,
    bn_decay = disc_bn_decay,               # no batch normalization for the critic
    critic = critic_bool,                   # Do you want a critic
    leaky_param = disc_leaky_param,         # parameter for leakyRelu
    mini_batch = mini_batch_bool,           # Do you want any mini batch extras
    add_rows = disc_add_rows # Number of rows to add if appending extra rows 
    )


optim_gen = optim.Adam(generator.parameters(),
                       weight_decay= gen_l2_regularization,
                       lr= gen_learning_rate
                       )

optim_disc = optim.Adam(discriminator.parameters(),
                        weight_decay= disc_l2_regularization,
                        lr= disc_learning_rate
                        )    

   

# option to load pretrained parameters. Make sure all of the parameters of the pretrained
# Model match up with the parameters you inilized above 

#generator.load_state_dict(torch.load('./training_iterations/test_start/generator_4_4050_50'))
#optim_gen.load_state_dict(torch.load('./training_iterations/test_start/gen_optim_3_3750_56'))
#discriminator.load_state_dict(torch.load('./training_iterations/test_start/discriminator_3_3850_16'))
#optim_disc.load_state_dict(torch.load('./training_iterations/test_start/disc_optim_3_3850_16'))


epochs = 1000
disc_epochs = 2
gen_epochs = 1
generator.train(mode=True)
discriminator.train(mode=True)
disc_losses = []
gen_losses = []
pois_metric = []
rf_metric = []
rf_imp_metric = []

disc_loss = torch.tensor(9999)
gen_loss = torch.tensor(9999)
loop = tqdm(total = epochs, position = 0, leave = False)
for epoch in range(epochs):
    for d_epoch in range(disc_epochs):  
        for c1,c2 in auto_loader: # c1 is continous variables and c2 is the categorical variables
            batch = torch.cat([c2,c1],1) 
            optim_disc.zero_grad()
            
            # train discriminator with real data
            real_features = Variable(batch)
            real_pred = discriminator(real_features)
            # the disc outputs high numbers if it thinks the data is real, we take the negative of this
            # Because we are minimizing loss
            real_loss = -real_pred.mean(0).view(1)  
            real_loss.backward()

            # then train the discriminator only with fake data
            noise = Variable(torch.FloatTensor(len(batch), z_size).normal_())
            fake_features = generator(noise, training = True)
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
            disc_losses = disc_loss.item()
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
        gen_features = generator(noise)
        gen_pred = discriminator(gen_features)

        gen_loss = - gen_pred.mean(0).view(1)
        gen_loss.backward()

        optim_gen.step()

        gen_loss = gen_loss
        gen_losses = gen_loss.item()
        del gen_loss 
        del noise
        del gen_features
        del gen_pred
            
    loop.set_description('epoch:{}, disc_loss:{:.4f}, gen_loss:{:.4f}'.format(epoch, disc_losses, gen_losses))
    loop.update(1) 
    # analyze poisson regression parameters every 20 epochs
    if(epoch % 1 == 0):
        with torch.no_grad():
            generated_data = generator(Variable(torch.FloatTensor(pol_dat.shape[0], z_size).normal_()), training = False)
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
        gen_features= X_gen
        gen_feature_list = list(gen_features.columns)
        gen_features = np.array(gen_features)
        y_gen2 = np.squeeze(y_gen)/np.squeeze(df2['Exposure'])
        rf_gen = RandomForestRegressor(n_estimators = 1000, random_state = 42)
        rf_gen.fit(gen_features, np.squeeze(y_gen))
        
        gen_predictions = rf_gen.predict(X_test)
        
        importances_gen = rf_gen.feature_importances_
        
        # Calculate Errors
        errors_pois = poisson_mod_gen.predict(X_test) - real_pois_preds
        errors_rf = abs(gen_predictions - real_predictions)
        errors_imp = abs(importances_gen - importances_real)
        
        pois_metric.append(round(np.mean(errors_rf), 4))
        rf_metric.append(round(np.mean(errors_pois), 4))
        rf_imp_metric.append(np.mean(errors_imp))
        
        if(epoch > 3) :
            plt.subplot(311)
            plt.plot(pois_metric, label = 'train')
            plt.ylabel('poission Dif')
            
            plt.subplot(312)
            plt.plot(rf_metric, label = 'train')
            plt.ylabel('rf pred dif')
            
            plt.subplot(313)
            plt.plot(rf_imp_metric, label = 'train')
            plt.ylabel('rf imp dif')
            #plt.savefig(output_fig_save_path, bbox_inches='tight')
            plt.show()
            plt.clf()
        
        print('Mean Absolute Difference RF:', round(np.mean(errors_rf), 2))
        print('Mean Absolute Difference Pois:', round(np.mean(errors_pois), 2))
        print('Mean Absolute Difference RF Imp:', round(np.mean(errors_imp), 2))
        
        del errors_pois
        del errors_rf
        del errors_imp
        del rf_gen
        del y_gen2
        del poisson_mod_gen
        del y_gen
        del X_gen
        del importances_gen
        del gen_predictions 
        del generated_data
        del df1
        del df2
        del gen_features
        
        
        #torch.save(generator.state_dict(), f='./saved_parameters/gen_test')
        #print(pois_df)

      
