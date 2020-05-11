#!/usr/bin/env python
# coding=utf-8

"""
Script to train the autoencoder with the MTPL data set for the auto-dpgan.
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
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

from config import AutoencoderConfig
from auto_dpgan.utils import prepare_data, mtpl_features
from auto_dpgan.dp_autoencoder import Autoencoder
from auto_dpgan import dp_optimizer, analysis, sampling

# Setup
sns.set()


def ae_training(x_train, x_test,
                iterations, log_step,
                ae, loss_fn,
                minib_loader, microb_loader,
                enc_optimizer, dec_optimizer,
                enc_scheduler, dec_scheduler):
    # Prepare variables
    train_loss, val_loss = (np.empty(iterations), np.empty(iterations))
    # It's faster to preallocate memory on big datasets vs appending

    # Build training log header
    fill = "-" * 10
    result_msg = (fill + f" Training - {iterations:d} iters " + fill + "\n")
    log_format = "{:<12}    {:<12}    {:<12}"
    result_msg += (log_format.format("Iteration", "Train Loss", "Val Loss")
                   + "\n")

    # gradient_norms = np.empty(iterations)  # to find l2 norm clipping value

    # Start training
    for iteration, X_minibatch in enumerate(minib_loader(x_train)):
        ae.train()
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()

        # Microbatch training for DP-SGD
        train_loss_iter = []
        # grad_norm = 0  # to find l2 norm clipping value
        for X_microbatch in microb_loader(X_minibatch):
            dec_optimizer.zero_microbatch_grad()
            output = ae(X_microbatch)
            train_loss_micro = loss_fn(output, X_microbatch)
            train_loss_micro.backward()
            train_loss_iter.append(train_loss_micro.item())
            dec_optimizer.microbatch_step()
            # grad_norm += dec_optimizer.microbatch_step()  # l2 norm clipping
        # gradient_norms[iteration] = grad_norm  # l2 norm clipping

        # Compute average of the microbatches losses
        train_loss_iter = np.mean(train_loss_iter).item()

        # Learn and improve
        enc_optimizer.step()
        dec_optimizer.step()

        # Validation
        ae.eval()
        with torch.no_grad():  # Faster not to compute gradients
            val_loss_iter = loss_fn(ae(x_test).detach(), x_test).item()

        # Store losses
        train_loss[iteration] = train_loss_iter
        val_loss[iteration] = val_loss_iter

        # Scheduler step
        dec_scheduler.step(val_loss_iter)
        enc_scheduler.step(val_loss_iter)

        # Log results at the given interval
        if (iteration + 1) % log_step == 0:
            print('[Iteration %d/%d] [Loss: %f] [Validation Loss: %f]' % (
                iteration + 1, iterations, train_loss_iter, val_loss_iter)
                  )
            result_msg += (log_format.format(
                f"{iteration + 1:d}",
                f"{train_loss_iter:.4f}",
                f"{val_loss_iter:.4f}") + "\n")

    # print(np.median(gradient_norms))  # l2 norm clipping

    return train_loss, val_loss, result_msg


def plot_loss_history(train_loss, val_loss, save_name):
    """Plot and save the train and validation loss history.

    @param train_loss: (list) Train loss for each iteration
    @param val_loss: (list) Validation loss for each iteration
    @param save_name: (str) Path where to save the plot
    @return: None but will save the plot as a png image
    """
    history = pd.DataFrame(data={'iter': list(range(len(train_loss))),
                                 'train': train_loss,
                                 'val': val_loss})

    plt.figure()
    sns.lineplot(x="iter", y="train", data=history, label="Train")
    sns.lineplot(x="iter", y="val", data=history, label="Validation")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.2e'))
    plt.xticks(rotation=30)
    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.title("Autoencoder loss history")
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches="tight")
    plt.close()


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
    parser.add_argument("--config", type=str, default="baseline",
                        help="name of the configuration to use")
    parser.add_argument("--iterations", type=int, default=20000,
                        help="number of training iterations")
    parser.add_argument("--save_model", action="store_true",  # default = False
                        help="will save the trained autoencoder")
    parser.add_argument("--random_seed", type=int, default=13,
                        help="fix random number generator sequence")
    parser.add_argument("--results_path", type=str,
                        default=Path.home() / f"mncdp/ae_{date_and_time}/",
                        help="path to folder where to save the results")
    parser.add_argument("--cpu_threads", type=int,
                        default=torch.get_num_threads(),
                        help="number of threads to run on if using CPU")

    # Differential privacy settings
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

    # Load desired configuration
    params = AutoencoderConfig(args.config)

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
                                  + str(Path("/train.log"))),
                        format='%(message)s',
                        filemode="w",  # Overwrite instead of appending
                        level=logging.DEBUG)

    # Disable matplotlib debugging logger
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)

    # Log the parser options
    logging.info("Options:")
    for name, value in vars(args).items():
        logging.info(f"\t{name} = {value}")
    logging.info("\n")

    # Log the configuration of the autoencoder
    logging.info("Configuration:")
    for name, value in vars(params).items():
        logging.info(f"\t{name} = {value}")
    logging.info("\n")

    # ----------------------------- Prepare data -----------------------------
    # Load and process data
    X_processed, map_of_types = prepare_data(
        Path(params.data_path).absolute(),
        mtpl_features + params.features
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

    # ------------------- Build model, optimizers and loss -------------------
    autoencoder = Autoencoder(
        example_dim=len(X_train[0]),
        compression_dim=params.compress_dim,
        feature_choice=params.features,
        binary=params.binary,
        device=device,
        init_weights=params.weights_init
    )

    decoder_optimizer = dp_optimizer.DPAdam(
        l2_norm_clip=params.l2_norm_clip,
        noise_multiplier=args.noise_multiplier,
        minibatch_size=params.minibatch_size,
        microbatch_size=params.microbatch_size,
        nonprivate=args.nonprivate,
        params=autoencoder.get_decoder().parameters(),
        lr=params.lr,
        betas=(params.b1, params.b2),
        weight_decay=params.l2_penalty,
    )

    encoder_optimizer = torch.optim.Adam(
        params=autoencoder.get_encoder().parameters(),
        lr=(params.lr * params.microbatch_size / params.minibatch_size),
        betas=(params.b1, params.b2),
        weight_decay=params.l2_penalty,
    )

    scheduler_factor = 0.2  # Factor by which to reduce the lr
    scheduler_threshold = 1e-4  # Threshold on the validation accuracy
    lr_minimum = params.lr / 100  # Limit minimum value

    decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        decoder_optimizer,
        factor=scheduler_factor,
        patience=params.patience,
        verbose=True,
        threshold=scheduler_threshold,
        min_lr=lr_minimum,
    )

    encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        encoder_optimizer,
        factor=scheduler_factor,
        patience=params.patience,
        verbose=True,
        threshold=scheduler_threshold,
        min_lr=(lr_minimum * params.microbatch_size / params.minibatch_size),
    )

    loss_function = torch.nn.BCELoss()  # Binary cross entropy

    # Log architecture of the autoencoder
    print(autoencoder)
    logging.info(autoencoder)
    logging.info("\n")

    # Check quality of differential privacy
    privacy_check = 'Achieves ({}, {})-DP'.format(
        analysis.epsilon(
            len(X_train),
            params.minibatch_size,
            args.noise_multiplier,
            args.iterations,
            args.delta),
        args.delta)
    print(privacy_check)
    logging.info("Privacy check:")
    logging.info(privacy_check + "\n")

    # Set data loaders
    minibatch_loader, microbatch_loader = sampling.get_data_loaders(
        minibatch_size=params.minibatch_size,
        microbatch_size=params.microbatch_size,
        iterations=args.iterations,
        nonprivate=args.nonprivate,
    )

    # -------------------------- Train autoencoder --------------------------
    train_start = time()  # To measure total training time

    # Faster to send all data to GPU than to use multithreads
    X_train = X_train.to(device)
    X_test = X_test.to(device)

    log_interval = args.iterations // 10

    # Training loop
    train_losses, validation_losses, result_message = ae_training(
        X_train, X_test,
        args.iterations, log_interval,
        autoencoder, loss_function,
        minibatch_loader, microbatch_loader,
        encoder_optimizer, decoder_optimizer,
        encoder_scheduler, decoder_scheduler
    )

    # Log the training history
    logging.info(result_message)

    # Save autoencoder model if desired
    if args.save_model:
        file_name = str(args.results_path) + str(Path("/ae_model.dat"))
        with open(file_name, 'wb') as f:
            torch.save(autoencoder, f)

    # Plot and save training history
    file_name = str(args.results_path) + str(Path("/loss.pkl"))
    pd.DataFrame(list(zip(train_losses, validation_losses)),
                 columns=['train_loss', 'validation_loss']
                 ).to_pickle(file_name)
    file_name = str(args.results_path) + str(Path("/loss.png"))
    plot_loss_history(train_losses, validation_losses, file_name)

    # Log runtime and train time
    total_train_time = f"\nTotal train time: {time() - train_start:.4f}s"
    total_runtime = f"Total runtime: {time() - start_time:.4f}s"
    print(total_train_time)
    print(total_runtime)
    logging.info(total_train_time)
    logging.info(total_runtime)

    print(f"\nResults were saved in the folder: {args.results_path}")
