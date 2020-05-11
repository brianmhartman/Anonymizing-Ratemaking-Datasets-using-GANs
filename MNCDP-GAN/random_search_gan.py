#!/usr/bin/env python
# coding=utf-8

"""
Script to tune the hyperparameters of the GAN using a Random Search.
"""

__author__ = "Olivier Mercier"

import argparse
import logging
from time import time
from datetime import datetime
from pathlib import Path
from random import seed

import torch
import numpy as np

from config import GanConfig
from auto_dpgan.utils import prepare_data, mtpl_features
from auto_dpgan.dp_wgan import Generator, Discriminator
from auto_dpgan import dp_optimizer, sampling
from train_gan import gan_training, plot_loss_history


if __name__ == "__main__":
    # --------------------------------- Time ---------------------------------
    start_time = time()  # to compute total runtime
    dt = datetime.now()
    date_and_time = (dt.strftime("%x").replace("/", "-") + "_"
                     + dt.strftime("%H") + "h"
                     + dt.strftime("%M") + "m"
                     + dt.strftime("%M") + "s")  # MM-DD-YY_HHhMMmSSs
    del dt

    # -------------------------------- Parser --------------------------------
    parser = argparse.ArgumentParser()

    # Paths and options
    parser.add_argument("--runs", type=int, default=50,
                        help="number of random searches to run")
    parser.add_argument("--iterations", type=int, default=20000,
                        help="number of training iterations")
    parser.add_argument("--ae_path", type=str,
                        default="./saved_models/baseline_ae.dat",
                        help="path to the pretrained autoencoder")
    parser.add_argument("--random_seed", type=int, default=13,
                        help="fix random number generator sequence")
    parser.add_argument("--save_loss_history", action="store_true",
                        help="will save the plot of the loss history "
                             "of each run as a png image")
    parser.add_argument("--results_path", type=str,
                        default=Path.home() / f"mncdp/rs_gan_{date_and_time}/",
                        help="path to folder where to save the results")
    parser.add_argument("--cpu_threads", type=int,
                        default=torch.get_num_threads(),
                        help="number of threads to run on if using CPU")

    # Random hyperparameters of the autoencoder
    parser.add_argument("--weights_init", type=str, default="both",
                        help="whether to initialize the weights of the "
                             "GAN or not: 'yes', 'no' or 'both'")
    parser.add_argument("--minibatch_size", nargs=2,  type=int,
                        default=[64, 64],
                        help="(min, max) size of each minibatch")
    parser.add_argument("--microbatch_size", nargs=2, type=int,
                        default=[1, 1],
                        help="(min, max) size of each microbatch (only for DP")
    parser.add_argument("--latent_dim", nargs=2,  type=int, default=[20, 30],
                        help="(min, max) latent dimensions in the generator")
    parser.add_argument("--lr", nargs=2,  type=float, default=[5e-6, 5e-5],
                        help="(min, max) value of the learning rate")
    parser.add_argument("--d_updates", nargs=2,  type=int, default=[5, 15],
                        help="(min, max) iterations of the discriminator "
                             "before updating the generator once")
    parser.add_argument("--optimizer", type=str, default="RMSprop",
                        help="possible optimizer of the GAN: "
                             "'Adam', 'RMSprop' or 'both'")
    parser.add_argument("--l2_penalty", nargs=2,  type=float,
                        default=[0., 0.001],
                        help="(min, max) value of the optimizer weight decay")
    parser.add_argument("--b1", nargs=2,  type=float, default=[0.9, 0.9],
                        help="(min, max) value of the beta 1 param for ADAM")
    parser.add_argument("--b2", nargs=2,  type=float, default=[0.999, 0.999],
                        help="(min, max) value of the beta 2 param for ADAM")
    parser.add_argument("--alpha", nargs=2,  type=float, default=[0.9, 0.999],
                        help="(min, max) smoothing constant for RMSProp")

    # Differential privacy settings (better to search with nonprivate)
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="value of delta for differential privacy")
    parser.add_argument("--noise_multiplier", type=float, default=3.5,
                        help="multiplication factor for the injected noise")
    parser.add_argument("--private", dest="nonprivate", action="store_false",
                        default=True,  # Nonprivate by default
                        help="will apply differential privacy")

    # Save choices as parser
    args = parser.parse_args()

    # -------------------------------- Setup --------------------------------
    # Set random seeds
    seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Load blank configuration
    params = GanConfig(name="random_search")

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # If using CPU, specify number of threads to run on
    if device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    
    # Build result directory if it doesn't already exist
    if not Path(args.results_path).exists():
        Path(args.results_path).mkdir(parents=True, exist_ok=True)

    # Build logger for results
    logging.basicConfig(filename=(str(args.results_path)
                                  + str(Path("/random_search.log"))),
                        format='%(message)s',
                        filemode="w",  # Overwrite instead of appending
                        level=logging.DEBUG)

    # Disable matplotlib debugging logger
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)

    # Log the configuration
    logging.info("Configuration:")
    for name, value in vars(params).items():
        logging.info(f"\t{name} = {value}")
    for name, value in vars(args).items():
        logging.info(f"\t{name} = {value}")
    logging.info("\n")

    # Load the pretrained autoencoder
    with open(Path(args.ae_path), 'rb') as f:
        if device == "cpu":
            autoencoder = torch.load(f, map_location="cpu")
        else:
            autoencoder = torch.load(f)
    decoder = autoencoder.get_decoder()

    # ----------------------------- Prepare data -----------------------------
    # Load and process data
    X_processed, map_of_types = prepare_data(
        Path(params.data_path).absolute(),
        mtpl_features + autoencoder.feature_choice,
    )
    X_processed = torch.from_numpy(X_processed)

    # Log the type of each feature
    logging.info("Type of each feature:")
    for feature_name, feature_type in map_of_types:
        logging.info(f"\t{feature_name} : {feature_type}")
    logging.info("\n")

    # Split train data (test is not needed for the random search of the GAN)
    train_cutoff = round(params.train_ratio * len(X_processed))
    X_train = X_processed[:train_cutoff]
    X_test = X_processed[train_cutoff:]

    # Send to GPU if possible
    X_train = X_train.to(device)
    X_test = X_test.to(device)

    # ------------------------ Select hyperparameters ------------------------
    # Weight initialization choices - Pick boolean from option chosen
    weights_init = [False] * args.runs
    if args.weights_init == "yes":
        weights_init = [True] * args.runs
    elif args.weights_init == "both":
        weights_init = np.random.choice((False, True), size=args.runs)

    # Minibatch size choices - Pick a power of 2 from given interval
    powers = np.log2(args.minibatch_size).astype(int)  # min/max
    powers = list(range(powers[0], powers[-1] + 1, 1))  # values in interval
    mb_possible_size = np.power(2, powers)
    minibatch_size = np.random.choice(mb_possible_size, size=args.runs)

    # Microbatch size choices - Pick a power of 2 from given interval
    powers = np.log2(args.microbatch_size).astype(int)  # min/max
    powers = list(range(powers[0], powers[-1] + 1, 1))  # values in interval
    mb_possible_size = np.power(2, powers)
    microbatch_size = np.random.choice(mb_possible_size, size=args.runs)

    # Compressed dimension choices - Pick int within given interval (inclusive)
    latent_dim = np.random.randint(args.latent_dim[0],
                                   args.latent_dim[-1] + 1,
                                   size=args.runs)

    # Learning rate choices - Random uniform from interval [low, high)
    lr = np.random.uniform(args.lr[0], args.lr[-1], size=args.runs)

    # Discriminator update choices - Pick int within given interval (inclusive)
    d_updates = np.random.randint(args.d_updates[0],
                                  args.d_updates[-1] + 1,
                                  size=args.runs)

    # Optimizer choices - Pick optimizer from option chosen
    optimizer = ["RMSprop"] * args.runs
    if args.optimizer == "Adam":
        optimizer = ["Adam"] * args.runs
    elif args.optimizer == "both":
        optimizer = np.random.choice(("RMSprop", "Adam"), size=args.runs)

    # Beta 1 choices - Random uniform from interval [low, high)
    b1 = np.random.uniform(args.b1[0], args.b1[-1], size=args.runs)

    # Beta 2 choices - Random uniform from interval [low, high)
    b2 = np.random.uniform(args.b2[0], args.b2[-1], size=args.runs)

    # l2 penalty choices - Random uniform from interval [low, high)
    l2_penalty = np.random.uniform(args.l2_penalty[0],
                                   args.l2_penalty[-1],
                                   size=args.runs)

    # Alpha choices - Random uniform from interval [low, high)
    alpha = np.random.uniform(args.alpha[0], args.alpha[-1], size=args.runs)

    # ---------------------------- Random Search ----------------------------
    log_interval = args.iterations // 10  # Iters step for logging results
    run_losses = np.empty((args.runs, 3))  # Store the final losses of each run
    fill = "-" * 25  # For run header

    for run in range(args.runs):
        header = fill + f" Run {run + 1} " + fill
        print(header)
        logging.info(header)

        # Log the chosen hyperparemeters for the current run
        choices = {"weights_init": weights_init[run],
                   "minibatch_size": minibatch_size[run],
                   "microbatch_size": microbatch_size[run],
                   "latent_dim": latent_dim[run],
                   "lr": lr[run],
                   "d_updates": d_updates[run],
                   "optimizer": optimizer[run],
                   "l2_penalty": l2_penalty[run],
                   }
        if optimizer[run] == "Adam":
            choices["b1"] = b1[run]
            choices["b2"] = b2[run]
        else:
            choices["alpha"] = alpha[run]

        logging.info("Hyperparameters:")
        for name, value in choices.items():
            logging.info(f"\t{name} = {value}")
        logging.info("\n")

        train_start = time()  # To measure total training time

        generator = Generator(
            input_dim=latent_dim[run],
            output_dim=autoencoder.get_compression_dim(),
            binary=params.binary,
            device=device,
            init_weights=weights_init[run]
        )

        discriminator = Discriminator(
            input_dim=len(X_train[0]),
            device=device,
            init_weights=weights_init[run]
        )

        if optimizer[run] == "RMSprop":
            generator_optimizer = torch.optim.RMSprop(
                params=generator.parameters(),
                lr=lr[run],
                alpha=alpha[run],
                weight_decay=l2_penalty[run],
            )

            discriminator_optimizer = dp_optimizer.DPRMSprop(
                l2_norm_clip=params.l2_norm_clip,
                noise_multiplier=args.noise_multiplier,
                minibatch_size=minibatch_size[run],
                microbatch_size=microbatch_size[run],
                nonprivate=args.nonprivate,
                params=discriminator.parameters(),
                lr=lr[run],
                alpha=alpha[run],
                weight_decay=l2_penalty[run],
            )

        else:  # Adam
            generator_optimizer = torch.optim.Adam(
                params=generator.parameters(),
                lr=lr[run],
                betas=(b1[run], b2[run]),
                weight_decay=l2_penalty[run],
            )

            discriminator_optimizer = dp_optimizer.DPAdam(
                l2_norm_clip=params.l2_norm_clip,
                noise_multiplier=args.noise_multiplier,
                minibatch_size=minibatch_size[run],
                microbatch_size=microbatch_size[run],
                nonprivate=args.nonprivate,
                params=discriminator.parameters(),
                lr=lr[run],
                betas=(b1[run], b2[run]),
                weight_decay=l2_penalty[run],
            )

        # Set data loaders
        minibatch_loader, microbatch_loader = sampling.get_data_loaders(
            minibatch_size=int(minibatch_size[run]),
            microbatch_size=microbatch_size[run],
            iterations=args.iterations,
            nonprivate=args.nonprivate,
        )

        # Training loop
        d_losses, g_losses, result_message = gan_training(
            X_train, X_test, args.iterations, log_interval,
            decoder, discriminator, generator,
            minibatch_loader, microbatch_loader,
            discriminator_optimizer, generator_optimizer,
            d_updates[run], latent_dim[run], params.clip_value,
            device,
        )

        # Log the training history
        logging.info(result_message)

        # Plot and save training history if desired
        if args.save_loss_history:
            file_name = (str(args.results_path) + str(Path(f"/run{run + 1}_")))
            plot_loss_history(d_losses, g_losses, file_name)

        # Log train time
        total_train_time = f"Total train time: {time() - train_start:.4f}s\n"
        print(total_train_time)
        logging.info(total_train_time)

    # Log runtime
    total_runtime = f"\nTotal runtime: {time() - start_time:.4f}s"
    print(total_runtime)
    logging.info(total_runtime)

    print(f"\nResults were saved in the folder: {args.results_path}")
