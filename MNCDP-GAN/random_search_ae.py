#!/usr/bin/env python
# coding=utf-8

"""
Script to tune the hyperparameters of the autoencoder using a Random Search.
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

from config import AutoencoderConfig
from auto_dpgan.utils import prepare_data, mtpl_features
from auto_dpgan.dp_autoencoder import Autoencoder
from auto_dpgan import dp_optimizer, sampling
from train_ae import ae_training, plot_loss_history


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
    parser.add_argument("--runs", type=int, default=100,
                        help="number of random searches to run")
    parser.add_argument("--random_seed", type=int, default=13,
                        help="fix random number generator sequence")
    parser.add_argument("--iterations", type=int, default=20000,
                        help="number of training iterations")
    parser.add_argument("--save_loss_history", action="store_true",
                        help="will save the plot of the loss history "
                             "of each run as a png image")
    parser.add_argument("--results_path", type=str,
                        default=Path.home() / f"mncdp/rs_ae_{date_and_time}/",
                        help="path to folder where to save the results")
    parser.add_argument("--cpu_threads", type=int,
                        default=torch.get_num_threads(),
                        help="number of threads to run on if using CPU")

    # Non random parameters
    parser.add_argument("--features", nargs=5, type=str,  # List of 5 args
                        default=["DriverAge", "CarAge",
                                 "Density", "Exposure", "ClaimNb_cat"],
                        help="types of 'DriverAge', 'CarAge', 'Density', "
                             "'Exposure' and 'ClaimNb' (bin or cat)")
    # Usage: --features DriverAge_bin CarAge Density_cat Exposure ClaimNb_cat

    # Random hyperparameters of the autoencoder
    parser.add_argument("--weights_init", type=str, default="both",
                        help="whether to initialize the weights of the "
                             "autoencoder or not: 'yes', 'no' or 'both'")
    parser.add_argument("--minibatch_size", nargs=2, type=int,
                        default=[64, 128],
                        help="(min, max) size of each minibatch")
    parser.add_argument("--microbatch_size", nargs=2, type=int,
                        default=[1, 1],
                        help="(min, max) size of each microbatch (only for DP")
    parser.add_argument("--compress_dim", nargs=2, type=int, default=[5, 30],
                        help="(min, max) number of dims of the latent space")
    parser.add_argument("--lr", nargs=2, type=float, default=[1e-3, 5e-2],
                        help="(min, max) initial value of the learning rate")
    parser.add_argument("--patience", nargs=2, type=int,
                        default=[20000, 20000],
                        help="(min, max) number of iterations before lowering "
                             "the lr if the validation loss doesn't improve")
    parser.add_argument("--b1", nargs=2, type=float, default=[0.9, 0.9],
                        help="(min, max) value of the beta 1 param for ADAM")
    parser.add_argument("--b2", nargs=2, type=float, default=[0.999, 0.999],
                        help="(min, max) value of the beta 2 param for ADAM")
    parser.add_argument("--l2_penalty", nargs=2, type=float, default=[0., 0.],
                        help="(min, max) value of the weight decay for ADAM")

    # Differential privacy settings (better to search with nonprivate)
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="value of delta for differential privacy")
    parser.add_argument("--noise_multiplier", type=float, default=2.5,
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
    params = AutoencoderConfig(name="random_search")

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

    # ----------------------------- Prepare data -----------------------------
    # Load and process data
    X_processed, map_of_types = prepare_data(
        Path(params.data_path).absolute(),
        mtpl_features + args.features
    )
    X_processed = torch.from_numpy(X_processed)

    # Log the type of each feature
    logging.info("Type of each feature:")
    for feature_name, feature_type in map_of_types:
        logging.info(f"\t{feature_name} : {feature_type}")
    logging.info("\n")

    # Split train/test data
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
    compress_dim = np.random.randint(args.compress_dim[0],
                                     args.compress_dim[-1] + 1,
                                     size=args.runs)

    # Learning rate choices - Random uniform from interval [low, high)
    lr = np.random.uniform(args.lr[0], args.lr[-1], size=args.runs)

    # Patience choices - Pick int within given interval (inclusive)
    patience = np.random.randint(args.patience[0],
                                 args.patience[-1] + 1,
                                 size=args.runs)

    # Beta 1 choices - Random uniform from interval [low, high)
    b1 = np.random.uniform(args.b1[0], args.b1[-1], size=args.runs)

    # Beta 2 choices - Random uniform from interval [low, high)
    b2 = np.random.uniform(args.b2[0], args.b2[-1], size=args.runs)

    # l2 penalty choices - Random uniform from interval [low, high)
    l2_penalty = np.random.uniform(args.l2_penalty[0],
                                   args.l2_penalty[-1],
                                   size=args.runs)

    # ---------------------------- Random Search ----------------------------
    log_interval = args.iterations // 10  # Iters step for logging results
    loss_function = torch.nn.BCELoss()  # Binary cross entropy
    run_losses = np.empty((args.runs, 2))  # To store the val loss of each run
    fill = "-" * 25  # For run header

    for run in range(args.runs):
        header = fill + f" Run {run + 1} " + fill
        print(header)
        logging.info(header)

        # Log the chosen hyperparemeters for the current run
        choices = {"weights_init": weights_init[run],
                   "minibatch_size": minibatch_size[run],
                   "microbatch_size": microbatch_size[run],
                   "compress_dim": compress_dim[run],
                   "lr": lr[run],
                   "patience": patience[run],
                   "b1": b1[run],
                   "b2": b2[run],
                   "l2_penalty": l2_penalty[run],
                   }
        logging.info("Hyperparameters:")
        for name, value in choices.items():
            logging.info(f"\t{name} = {value}")
        logging.info("\n")

        train_start = time()  # To measure total training time

        autoencoder = Autoencoder(
            example_dim=len(X_train[0]),
            compression_dim=compress_dim[run],
            feature_choice=args.features,
            binary=params.binary,
            device=device,
            init_weights=weights_init[run]
        )

        decoder_optimizer = dp_optimizer.DPAdam(
            l2_norm_clip=params.l2_norm_clip,
            noise_multiplier=args.noise_multiplier,
            minibatch_size=minibatch_size[run],
            microbatch_size=microbatch_size[run],
            nonprivate=args.nonprivate,
            params=autoencoder.get_decoder().parameters(),
            lr=lr[run],
            betas=(b1[run], b2[run]),
            weight_decay=l2_penalty[run],
        )

        encoder_optimizer = torch.optim.Adam(
            params=autoencoder.get_encoder().parameters(),
            lr=(lr[run] * microbatch_size[run] / minibatch_size[run]),
            betas=(b1[run], b2[run]),
            weight_decay=l2_penalty[run],
        )

        scheduler_factor = 0.1  # Factor by which to reduce the lr
        scheduler_threshold = 1e-4  # Threshold on the validation accuracy
        lr_minimum = lr[run] / 100  # Limit minimum value

        decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            decoder_optimizer,
            factor=scheduler_factor,
            patience=patience[run],
            verbose=True,
            threshold=scheduler_threshold,
            min_lr=lr_minimum,
        )

        encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            encoder_optimizer,
            factor=scheduler_factor,
            patience=patience[run],
            verbose=True,
            threshold=scheduler_threshold,
            min_lr=(lr_minimum * microbatch_size[run] / minibatch_size[run]),
        )

        # Set data loaders
        minibatch_loader, microbatch_loader = sampling.get_data_loaders(
            minibatch_size=int(minibatch_size[run]),
            microbatch_size=microbatch_size[run],
            iterations=args.iterations,
            nonprivate=args.nonprivate,
        )

        # Training loop
        train_losses, validation_losses, result_message = ae_training(
            X_train, X_test,
            args.iterations, log_interval,
            autoencoder, loss_function,
            minibatch_loader, microbatch_loader,
            encoder_optimizer, decoder_optimizer,
            encoder_scheduler, decoder_scheduler
        )

        # Store validation loss of last iteration
        run_losses[run] = [run, validation_losses[-1]]

        # Log the training history
        logging.info(result_message)

        # Plot and save training history if desired
        if args.save_loss_history:
            file_name = (str(args.results_path)
                         + str(Path(f"/loss_run_{run + 1}.png")))
            plot_loss_history(train_losses, validation_losses, file_name)

        # Log train time
        total_train_time = f"Total train time: {time() - train_start:.4f}s\n"
        print(total_train_time)
        logging.info(total_train_time)

    # Log summary of results
    logging.info(fill + " Summary " + fill)

    # Log the runs by ascending validation loss of the last iteration
    log_format = "{:<8}    {:<16}"
    logging.info(log_format.format("Run", "Last Val Loss"))
    run_losses = run_losses[np.argsort(run_losses[:, 1])]
    for run, loss in run_losses:
        logging.info(log_format.format(f"{int(run + 1):d}", f"{loss:.4f}"))

    # Log runtime
    total_runtime = f"\nTotal runtime: {time() - start_time:.4f}s"
    print(total_runtime)
    logging.info(total_runtime)

    print(f"\nResults were saved in the folder: {args.results_path}")
