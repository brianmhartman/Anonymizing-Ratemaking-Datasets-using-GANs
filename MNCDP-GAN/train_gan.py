#!/usr/bin/env python
# coding=utf-8

"""
Script to train the GAN with the MTPL data set for the auto-dpgan.
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

from config import GanConfig
from auto_dpgan.utils import prepare_data, mtpl_features
from auto_dpgan.dp_wgan import Generator, Discriminator
from auto_dpgan import dp_optimizer, analysis, sampling

# Setup
sns.set()


def gan_training(x_train, x_test,
                 iterations, log_step,
                 dec, discri, gen,
                 minib_loader, microb_loader,
                 d_optimizer, g_optimizer,
                 d_updates, latent_dim, clip_value,
                 _device, save_model=False, save_step=0):
    # Prepare variables
    train_g_loss = 0
    d_loss, g_loss = (np.empty((iterations, 2)), np.empty((iterations, 2)))
    # It's faster to preallocate memory on big datasets vs appending
    if not save_step:  # Evaluates to True if the value is 0
        save_step = log_step

    # Build training log header
    fill = "-" * 24
    result_msg = (fill + f" Training - {iterations:d} iters " + fill + "\n")
    log_format = "{:<12}    {:<12}    {:<12}    {:<12}    {:<12}"
    result_msg += log_format.format("Iteration",
                                    "Train D Loss", "Val D Loss",
                                    "Train G Loss", "Val G Loss")
    print()
    print(result_msg)
    result_msg += "\n"

    # gradient_norms = np.empty(iterations)  # to find l2 norm clipping value

    # Start training
    for iteration, (X_mini_train, X_mini_test) in \
            enumerate(zip(minib_loader(x_train), minib_loader(x_test))):

        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

        # Microbatch training of the discriminator for DP-SGD
        iter_d_loss = []
        # grad_norm = 0  # to find l2 norm clipping value
        discri.train()
        gen.eval()
        for real in microb_loader(X_mini_train):
            z = torch.randn(real.size(0),
                            latent_dim,
                            device=_device)
            fake = dec(gen(z)).detach()  # Detach = no differentiation of grad

            d_optimizer.zero_microbatch_grad()
            # Google WGAN -> Critic loss to MAXIMIZE: D(real) - D(fake)
            # Since the optimizer is MINIMIZING the loss, we invert the signs
            microbatch_d_loss = (-torch.mean(discri(real))
                                 + torch.mean(discri(fake)))
            # Discriminator predicts a score of how real or fake a given input
            # looks. The bigger the value, the more real he thinks it is. The
            # smaller (can be negative), the more fake. It is not bounded.
            microbatch_d_loss.backward()
            iter_d_loss.append(microbatch_d_loss.item())
            d_optimizer.microbatch_step()
            # grad_norm += d_optimizer.microbatch_step()  # l2 norm clipping
        # gradient_norms[iteration] = grad_norm  # l2 norm clipping

        d_optimizer.step()

        # Compute average of the microbatches losses
        train_d_loss = np.mean(iter_d_loss).item()

        # WGANs require clipping the gradients of the discriminator
        for parameter in discri.parameters():
            parameter.data.clamp_(-clip_value, clip_value)

        # Discriminator validation
        discri.eval()
        with torch.no_grad():
            z = torch.randn(X_mini_test.size(0),
                            latent_dim,
                            device=_device)
            fake = dec(gen(z)).detach()
            val_d_loss = (-torch.mean(discri(X_mini_test))
                          + torch.mean(discri(fake))).item()

        # Generator update
        # In WGANs, the discriminator must be updated more than the generator
        if iteration % d_updates == 0:  # This gets executed at iter 0
            gen.train()

            z = torch.randn(X_mini_train.size(0),
                            latent_dim,
                            device=_device)
            fake = dec(gen(z))

            g_optimizer.zero_grad()
            # Google WGAN -> Generator loss to MAXIMIZE: D(fake)
            train_g_loss = -torch.mean(discri(fake))
            train_g_loss.backward()
            train_g_loss = train_g_loss.item()
            g_optimizer.step()

            # Generator validation
            gen.eval()
            with torch.no_grad():
                z = torch.randn(X_mini_test.size(0),
                                latent_dim,
                                device=_device)
                fake = dec(gen(z))
                val_g_loss = -torch.mean(discri(fake)).item()

        # Log results at the given interval
        # The loss of the generator is the one from the last update
        if (iteration + 1) % log_step == 0:
            results = log_format.format(
                f"{iteration + 1:d}",
                f"{train_d_loss:.2e}", f"{val_d_loss:.2e}",
                f"{train_g_loss:.3e}", f"{val_g_loss:.3e}")
            print(results)
            result_msg += results
            result_msg += "\n"

        # Save relevant stuff at the given interval (if chosen)
        if save_model:
            if (iteration + 1) % save_step == 0:
                # Save GAN model
                save_name = str(args.results_path) + str(
                    Path(f"/gan_model_{iteration + 1}.dat"))
                with open(save_name, 'wb') as file:
                    torch.save(gen, file)

        # Save losses
        d_loss[iteration, 0] = train_d_loss  # Average of the microbatches
        d_loss[iteration, 1] = val_d_loss
        g_loss[iteration, 0] = train_g_loss  # The one from the last update
        g_loss[iteration, 1] = val_g_loss
        # For the gen, the same loss should repeat d_updates times

    # print(np.median(gradient_norms))  # l2 norm clipping

    return d_loss, g_loss, result_msg


def plot_loss_history(d_loss, g_loss, save_name):
    """Plot and save the train and validation loss history for the
    discriminator and the generator.

    @param d_loss: (list) Discriminator losses for each iteration (I x 2)
    @param g_loss: (list) Generator losses for each iteration (I x 2)
    @param save_name: (str) Path where to save the plot
    @return: None but will save the plot as a png image
    """
    # Format data for easy plotting
    history = pd.DataFrame(data={'iter': [x for x in list(range(len(d_loss)))],
                                 'd_train': d_loss[:, 0],
                                 'd_val': d_loss[:, 1],
                                 'g_train': g_loss[:, 0],
                                 'g_val': g_loss[:, 1]})

    # Discriminator loss plot
    plt.figure()
    sns.lineplot(x="iter", y="d_train", data=history, label="Train")
    sns.lineplot(x="iter", y="d_val", data=history, label="Validation")
    # Adjust y axis to see the small variations
    limit = 1.1 * np.max(abs(d_loss[20:, :])).item()
    plt.ylim(top=limit, bottom=-limit)
    # Format plot
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.2e'))
    plt.xticks(rotation=30)
    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.title("Discriminator loss history")
    plt.tight_layout()
    plt.savefig(save_name + str("d_loss.png"), bbox_inches="tight")
    # plt.show()  # For debug
    plt.close()

    # Generator loss plot
    plt.figure()
    sns.lineplot(x="iter", y="g_train", data=history, label="Train")
    sns.lineplot(x="iter", y="g_val", data=history, label="Validation")
    # Format plot
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.2e'))
    plt.xticks(rotation=30)
    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.title("Generator loss history")
    plt.tight_layout()
    plt.savefig(save_name + str("g_loss.png"), bbox_inches="tight")
    # plt.show()  # For debug
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
    parser.add_argument("--iterations", type=int, default=2e6,
                        help="number of training iterations")
    parser.add_argument("--save_model", action="store_true",  # default = False
                        help="will save the trained GAN")
    parser.add_argument("--save_step", type=int, default=0,
                        help="step for saving the model (in iterations). "
                             "If 0, will be set to 1/10th of iterations")
    parser.add_argument("--random_seed", type=int, default=13,
                        help="fix random number generator sequence")
    parser.add_argument("--results_path", type=str,
                        default=Path.home() / f"mncdp/gan_{date_and_time}/",
                        help="path to folder where to save the results")
    parser.add_argument("--cpu_threads", type=int,
                        default=torch.get_num_threads(),
                        help="number of threads to run on if using CPU")
    
    # Differential privacy settings
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

    # Load desired configuration
    params = GanConfig(args.config)

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

    # Load the pretrained autoencoder
    with open(Path(params.ae_path), 'rb') as f:
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

    # Split train data
    train_cutoff = round(params.train_ratio * len(X_processed))
    X_train = X_processed[:train_cutoff]
    X_test = X_processed[train_cutoff:]

    # ----------- Model, optimizers, schedulers, loss and loaders -----------
    generator = Generator(
        input_dim=params.latent_dim,
        output_dim=autoencoder.get_compression_dim(),
        binary=params.binary,
        device=device,
        init_weights=params.weights_init
    )

    discriminator = Discriminator(
        input_dim=len(X_train[0]),
        device=device,
        init_weights=params.weights_init
    )

    if params.optimizer == "RMSprop":
        generator_optimizer = torch.optim.RMSprop(
            params=generator.parameters(),
            lr=params.lr,
            alpha=params.alpha,
            weight_decay=params.l2_penalty,
        )

        discriminator_optimizer = dp_optimizer.DPRMSprop(
            l2_norm_clip=params.l2_norm_clip,
            noise_multiplier=args.noise_multiplier,
            minibatch_size=params.minibatch_size,
            microbatch_size=params.microbatch_size,
            nonprivate=args.nonprivate,
            params=discriminator.parameters(),
            lr=params.lr,
            alpha=params.alpha,
            weight_decay=params.l2_penalty,
        )
    else:
        generator_optimizer = torch.optim.Adam(
            params=generator.parameters(),
            lr=params.lr,
            betas=(params.b1, params.b2),
            weight_decay=params.l2_penalty,
        )

        discriminator_optimizer = dp_optimizer.DPAdam(
            l2_norm_clip=params.l2_norm_clip,
            noise_multiplier=args.noise_multiplier,
            minibatch_size=params.minibatch_size,
            microbatch_size=params.microbatch_size,
            nonprivate=args.nonprivate,
            params=discriminator.parameters(),
            lr=params.lr,
            betas=(params.b1, params.b2),
            weight_decay=params.l2_penalty,
        )

    # Log architecture of the GAN
    print(generator)
    print(discriminator)
    logging.info(generator)
    logging.info(discriminator)
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

    # ------------------------------ Train GAN ------------------------------
    train_start = time()  # To measure total training time

    # Faster to send all data to GPU than to use multithreads
    X_train = X_train.to(device)
    X_test = X_test.to(device)

    log_interval = args.iterations // 10

    # Training loop
    d_losses, g_losses, result_message = gan_training(
        X_train, X_test,
        args.iterations, log_interval,
        decoder, discriminator, generator,
        minibatch_loader, microbatch_loader,
        discriminator_optimizer, generator_optimizer,
        params.d_updates, params.latent_dim, params.clip_value,
        device,
        save_model=args.save_model,
        save_step=args.save_step,
    )

    # Log the training history
    logging.info(result_message)

    # Plot and save training history
    file_name = str(args.results_path) + str(Path("/loss.pkl"))
    pd.DataFrame.from_dict({"train_d_loss": d_losses[:, 0],
                            "val_d_loss": d_losses[:, 1],
                            "train_g_loss": g_losses[:, 0],
                            "val_g_loss": g_losses[:, 1]}
                           ).to_pickle(file_name)
    file_name = str(args.results_path) + str(Path("/"))
    plot_loss_history(d_losses, g_losses, file_name)

    # Log runtime and train time
    total_train_time = f"\nTotal train time: {time() - train_start:.4f}s"
    total_runtime = f"Total runtime: {time() - start_time:.4f}s"
    print(total_train_time)
    print(total_runtime)
    logging.info(total_train_time)
    logging.info(total_runtime)

    print(f"\nResults were saved in the folder: {args.results_path}")
