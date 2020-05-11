#!/usr/bin/env python
# coding=utf-8

"""
Configuration file for the autoencoder and GAN training.
"""


class Config:
    def __init__(self):
        self.data_path = "./data/preprocessed_MTPL.csv"
        self.train_ratio = 2/3
        self.binary = False  # LeakyReLu if False, tanh if True


class AutoencoderConfig(Config):
    def __init__(self, name="baseline"):
        super().__init__()
        self.l2_norm_clip = 0.022  # Determined empirically for MTPL1

        if name == "baseline":
            # Determined after Random Search and fine tuning
            self.features = ["DriverAge", "CarAge",
                             "Density", "Exposure",
                             "ClaimNb_cat"]
            self.weights_init = False
            self.minibatch_size = 64
            self.microbatch_size = 1
            self.compress_dim = 25  # number of dimensions of the latent space
            self.lr = 0.01  # initial value of the lr
            self.patience = 1000  # scheduler's patience (in iterations)
            self.b1 = 0.9  # beta 1 param for ADAM optimizer
            self.b2 = 0.999  # beta 2 param for ADAM optimizer
            self.l2_penalty = 0.  # weight decay of the optimizer

        elif name == "private64_baseline":
            # Determined after Random Search and fine tuning
            self.features = ["DriverAge", "CarAge",
                             "Density", "Exposure",
                             "ClaimNb_cat"]
            self.weights_init = False
            self.minibatch_size = 64
            self.microbatch_size = 64
            self.compress_dim = 25  # number of dimensions of the latent space
            self.lr = 0.01  # initial value of the lr
            self.patience = 1000  # scheduler's patience (in iterations)
            self.b1 = 0.9  # beta 1 param for ADAM optimizer
            self.b2 = 0.999  # beta 2 param for ADAM optimizer
            self.l2_penalty = 0.  # weight decay of the optimizer

        elif name == "all_cat":
            # Same hyperparameters as baseline (except the types of the vars)
            self.features = ["DriverAge", "CarAge",
                             "Density_cat", "Exposure_cat",
                             "ClaimNb_cat"]
            self.weights_init = False
            self.minibatch_size = 64
            self.microbatch_size = 1
            self.compress_dim = 25  # number of dimensions of the latent space
            self.lr = 0.01  # initial value of the lr
            self.patience = 1000  # scheduler's patience (in iterations)
            self.b1 = 0.9  # beta 1 param for ADAM optimizer
            self.b2 = 0.999  # beta 2 param for ADAM optimizer
            self.l2_penalty = 0.  # weight decay of the optimizer

        elif name == "bin":
            # Same hyperparameters as baseline (except the types of the vars)
            self.features = ["DriverAge_bin", "CarAge_bin",
                             "Log(Density)_bin", "Exposure_bin",
                             "ClaimNb_cat"]
            self.weights_init = False
            self.minibatch_size = 64
            self.microbatch_size = 1
            self.compress_dim = 25  # number of dimensions of the latent space
            self.lr = 0.01  # initial value of the lr
            self.patience = 1000  # scheduler's patience (in iterations)
            self.b1 = 0.9  # beta 1 param for ADAM optimizer
            self.b2 = 0.999  # beta 2 param for ADAM optimizer
            self.l2_penalty = 0.  # weight decay of the optimizer

        elif name == "tuned_bin":
            # Determined after Random Search and fine tuning
            self.features = ["DriverAge_bin", "CarAge_bin",
                             "Log(Density)_bin", "Exposure_bin",
                             "ClaimNb_cat"]
            self.weights_init = False
            self.minibatch_size = 128
            self.microbatch_size = 1
            self.compress_dim = 50  # number of dimensions of the latent space
            self.lr = 0.01  # initial value of the lr
            self.patience = 1000  # scheduler's patience (in iterations)
            self.b1 = 0.9  # beta 1 param for ADAM optimizer
            self.b2 = 0.999  # beta 2 param for ADAM optimizer
            self.l2_penalty = 0.  # weight decay of the optimizer


class GanConfig(Config):
    def __init__(self, name="baseline"):
        super().__init__()
        self.l2_norm_clip = 0.027  # Determined empirically for MTPL1
        self.clip_value = 0.01  # WGAN gradient clipping

        if name == "baseline":
            # Determined after Random Search and fine tuning
            self.ae_path = "./saved_models/baseline_ae.dat"
            self.features = ["DriverAge", "CarAge",
                             "Density", "Exposure",
                             "ClaimNb_cat"]
            self.weights_init = False
            self.minibatch_size = 128
            self.microbatch_size = 1
            self.latent_dim = 25  # latent dimensions in the generator
            self.lr = 4.5e-5  # learning rate (remains constant)
            self.d_updates = 10  # updates of the discri before 1 of the gen
            self.optimizer = "RMSprop"
            self.alpha = 0.99  # smoothing constant for the RMSProp optimizer
            self.b1 = None
            self.b2 = None
            self.l2_penalty = 0.  # weight decay of the optimizer

        elif name == "private64_baseline":
            # Determined after Random Search and fine tuning
            self.ae_path = "./saved_models/private64_baseline_ae.dat"
            self.features = ["DriverAge", "CarAge",
                             "Density", "Exposure",
                             "ClaimNb_cat"]
            self.weights_init = False
            self.minibatch_size = 128
            self.microbatch_size = 64
            self.latent_dim = 25  # latent dimensions in the generator
            self.lr = 4.5e-5  # learning rate (remains constant)
            self.d_updates = 10  # updates of the discri before 1 of the gen
            self.optimizer = "RMSprop"
            self.alpha = 0.99  # smoothing constant for the RMSProp optimizer
            self.b1 = None
            self.b2 = None
            self.l2_penalty = 0.  # weight decay of the optimizer

        elif name == "all_cat":
            # Same hyperparameters as baseline (except the types of the vars)
            self.ae_path = "./saved_models/all_cat_ae.dat"
            self.features = ["DriverAge", "CarAge",
                             "Density_cat", "Exposure_cat",
                             "ClaimNb_cat"]
            self.weights_init = False
            self.minibatch_size = 128
            self.microbatch_size = 1
            self.latent_dim = 25  # latent dimensions in the generator
            self.lr = 4.5e-5  # learning rate (remains constant)
            self.d_updates = 10  # updates of the discri before 1 of the gen
            self.optimizer = "RMSprop"
            self.alpha = 0.99  # smoothing constant for the RMSProp optimizer
            self.b1 = None
            self.b2 = None
            self.l2_penalty = 0.  # weight decay of the optimizer

        elif name == "bin":
            # Same hyperparameters as baseline (except the types of the vars)
            self.ae_path = "./saved_models/bin_ae.dat"
            self.features = ["DriverAge_bin", "CarAge_bin",
                             "Log(Density)_bin", "Exposure_bin",
                             "ClaimNb_cat"]
            self.weights_init = False
            self.minibatch_size = 128
            self.microbatch_size = 1
            self.latent_dim = 25  # latent dimensions in the generator
            self.lr = 4.5e-5  # learning rate (remains constant)
            self.d_updates = 10  # updates of the discri before 1 of the gen
            self.optimizer = "RMSprop"
            self.alpha = 0.99  # smoothing constant for the RMSProp optimizer
            self.b1 = None
            self.b2 = None
            self.l2_penalty = 0.  # weight decay of the optimizer

        elif name == "tuned_bin":
            # Determined after Random Search and fine tuning
            self.ae_path = "./saved_models/tuned_bin_ae.dat"
            self.features = ["DriverAge_bin", "CarAge_bin",
                             "Log(Density)_bin", "Exposure_bin",
                             "ClaimNb_cat"]
            self.weights_init = False
            self.minibatch_size = 128
            self.microbatch_size = 1
            self.latent_dim = 30  # latent dimensions in the generator
            self.lr = 3.9e-5  # learning rate (remains constant)
            self.d_updates = 5  # updates of the discri before 1 of the gen
            self.optimizer = "RMSprop"
            self.alpha = 0.99  # smoothing constant for the RMSProp optimizer
            self.b1 = None
            self.b2 = None
            self.l2_penalty = 0.  # weight decay of the optimizer

        elif name == "bad":
            # Configuration that appeared bad during random search
            self.ae_path = "./saved_models/baseline_ae.dat"
            self.features = ["DriverAge", "CarAge",
                             "Density", "Exposure",
                             "ClaimNb_cat"]
            self.weights_init = False
            self.minibatch_size = 128
            self.microbatch_size = 1
            self.latent_dim = 25  # latent dimensions in the generator
            self.lr = 5e-4  # learning rate (remains constant)
            self.d_updates = 10  # updates of the discri before 1 of the gen
            self.optimizer = "RMSprop"
            self.alpha = 0.99  # smoothing constant for the RMSProp optimizer
            self.b1 = None
            self.b2 = None
            self.l2_penalty = 0.  # weight decay of the optimizer

        elif name == "private_baseline":
            # Same as baseline, only the path to the AE is different
            self.ae_path = "./saved_models/private_baseline_ae.dat"
            self.features = ["DriverAge", "CarAge",
                             "Density", "Exposure",
                             "ClaimNb_cat"]
            self.weights_init = False
            self.minibatch_size = 128
            self.microbatch_size = 1  # Recommended for DP-SGD
            self.latent_dim = 25  # latent dimensions in the generator
            self.lr = 4.5e-5  # learning rate (remains constant)
            self.d_updates = 10  # updates of the discri before 1 of the gen
            self.optimizer = "RMSprop"
            self.alpha = 0.99  # smoothing constant for the RMSProp optimizer
            self.b1 = None
            self.b2 = None
            self.l2_penalty = 0.  # weight decay of the optimizer
