# MC-WGAN-GP

## GANs
This repository contains the code for two GAN setup. I refer to the GANs as the single GAN and the Multi GAN.
The single GAN makes use of an autoencoder where as the multi GAN setup does not. 



## auto_gan Folder
The auto_gan folder contains the files needed to train the GAN. 
1. tester_multicategorical.py : this script is used to walk through the multi GAN
2. tester_singleoutput.py: this script is used to walk through the single GAN
3. train_autoencoder.py: this script is used to train the autoencoder for the single GAN
4. gan_trainer_multi_v1.py: this script is used to train the multi GAN. It is meant to be called on the command line
5. gan_trainer_single_v1.py: this script is used to train the single GAN. It is meant to be called on the command line
6. gan_runner.R : this script is used to run multiple jobs at once
   -future work includes defining tuning parameters in R instead of Excel. 

## Sub Folders

1. Dataset: The datasets folder contains the data used to train the GAN
2. gan_scripts: The gan_scripts folder contains all of the scripts used to create the GAN
3. saved_parameters: The saved_parameters folder is where the parameters of trained models are stored to load them later
4. training_iterations: The training_iterations contains records of the different runs that have been completed and contains two folders.
   -4a. 19xxxx_iterations: This folder contains the structure to store model output
   -4b. iteration_records: This folder contains records of runs that have been done
   
## Process to train GAN on laptop
Simple run the tester_multicategorical.py or the tester_singleoutput.py files. 

## Process to Train GAN on server
1. Copy the auto_gan folder onto the server. 
2. Edit the flexible_template.csv file in ./training_iterations/iteration_records
   - note that this csv file has a column for the autoencoder location. If you are training the multi GAN you can leave this blank (or set to any value). 
3. Edit the gan_runner.R script to match your csv file and the script you want to train. 
4. Run the gan_runner.R script on the server

