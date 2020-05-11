# MNCDP-GAN
Mixed Numerical and Categorical and Differentially Private Generative Adversarial Network

As of 2020-04-29, in order to have no conflicts in the required packages (see requirements.txt), the Python version of your environment needs to be 3.7.

## Structure of the folder

###	MNCDP-GAN/

	- README

	- requirements.txt : List of packages required to run the code

	- config.py : Options and hyperparameters for the different configurations

	- mtpl_preprocessing.py : Preprocessing script of the MTPL data (does not include MTPL2)

	- train_ae.py : Training script of the autoencoder

	- train_gan.py : Training script of the GAN

	- random_search_ae.py : Tuning of the hyperparameters of the autoencoder using a random search

	- random_search_gan.py : Tuning of the hyperparameters of the GAN using a random search

	- test.py : Evaluation script of the trained model (requires autoencoder and generator)

####	auto_dpgan/
		
	Auto DP GAN module containing all the code files required to run the scripts
	
####	data/
		
	- OriginalMTPLData.csv : Unmodified original freMTPL dataset from CAS
	
	- preprocessed_MTPL.csv : Preprocessed version of the MTPL data from mtpl_preprocessing.py (used for training)
	
####	saved_models/
	
	- all_cat_ae.dat : Trained autoencoder (20K iterations) when Exposure and Density are categorical (from policy_dat_v2.csv)
	
	- all_cat_ae.dat : Trained GAN (2M iterations) when Exposure and Density are categorical (from policy_dat_v2.csv)
	
	- baseline_ae.dat : Trained autoencoder (20K iterations) with all features' types as in the original data
	
	- baseline_gan.dat : Trained GAN (2M iterations) with all features' types as in the original data
	
	- bin_ae.dat : Trained autoencoder (20K iterations) with binned CarAge, DriverAge, Exposure and Density
	
	- bin_gan.dat : Trained GAN (2M iterations) with binned CarAge, DriverAge, Exposure and Density

    - tuned_bin_ae.dat : Trained autoencoder (20K iterations) with binned CarAge, DriverAge, Exposure and Density after fine-tuning the hyperparameters using a random search

    - tuned_bin_gan.dat : Trained GAN (2M iterations) with binned CarAge, DriverAge, Exposure and Density after fine-tuning the hyperparameters using a random search

## Workflow
Starting from the original data, here's the workflow recommended to obtain a trained DP-auto-GAN model:
1. Run the mtpl_preprocessing.py script to obtain the preprocessed data (if not already available);
2. Using the random_search_ae.py script, narrow the range of the best hyperparameters for the autoencoder to a few possible configurations;
3. Update the config.py file with one of those good configurations;
4. Run the train_ae.py script using that configuration to check the final validation loss (don't forget to save the trained model and to keep the results);
5. Repeat steps 3 and 4 for the other good configurations found in the random search to find the best configuration (i.e. fine tuning) of the autoencoder;
6. Update the config.py file with the best fine tuned configuration of the autoencoder and put the corresponding saved model (.dat file) in the "saved_models" folder of the repo;
7. Using the random_search_gan.py scipt with the configuration of the autoencoder found at step 6, narrow the range of the best hyperparameters for the GAN to a few possible configurations;
8. Update the config.py file with one of those good configurations;
9. Run the train_ae.py script using that configuration and save the model;
10. Run the test.py script using that GAN model and the best autoencoder found at step 6 in order to evaluate the performance of the complete model (keep the results);
11. Repeat steps 8 to 10 for the other good configurations found in the random search to find the best configuration of the GAN;
12. Update the config.py file with the best fine tuned configuration of the GAN and put the corresponding saved model in the "saved_models" folder of the repo;
13. You now have a complete, fine-tuned and trained DP-auto-GAN model. You should use the same configuration to train the differentially private versions.


## Relevant papers for a better understanding of the code
Here's a list of recommended papers to read to grasp the code implementation of the DP-auto-GAN. Following the reference of each paper you have a brief description of what they cover in the code.
- Tantipongpipat et al. 2019, Differentially Private Mixed-Type Data Generation For Unsupervised Learning : Source of the base code (most of the folder auto_dpgan), algorithmic framework of the DPautoGAN and results on CENSUS data.
- Arjovsky et al. 2017, Wasserstein GAN : Basics of WGAN, Earth-Mover loss function, weight clamping of the discriminator, updating the discriminator (in this case called a "critic") more than the generator to train till optimality (prevents mode collapse), linear activation in the output layer of the discriminator instead of an activation function such as sigmoid, RMSProp optimizer and importance of learning rate for loss curves.
- Gulrajani et al. 2017, Improved Training of Wasserstein GANs : Improvements of WGAN, gradient penalty with Adam optimizer and choosing layer normalization over batch normalization for the generator.
- Abadi et al. 2016, Deep Learning with Differential Privacy : Differentially private stochastic gradient descent (DP-SGD) algorithm, privacy accountant and L2 norm clipping of the gradients.
- GANs section of the Google Machine Learning crash course (https://developers.google.com/machine-learning/gan) : Global explanation of GANs - see the chapter on "Loss Functions" for the implementation of the Earth-Mover distance for the discriminator and generator losses.


## Contributors
- Marie-Pier Côté
- Brian Hartman
- Jared Cummings
- Olivier Mercier
