#!/usr/bin/env python
# coding=utf-8

"""
Script to test the generated dataset of the auto-dpgan
"""

__author__ = "Olivier Mercier"

import argparse
import logging
from datetime import datetime
from pathlib import Path
from random import seed

import torch
import numpy as np
import pandas as pd

from auto_dpgan.utils import prepare_data, mtpl_features, \
    get_processed_feature_names
from auto_dpgan.evaluation import plot_categorical, plot_numeric, \
    rf_classification, rf_regression


if __name__ == "__main__":
    # --------------------------------- Date ---------------------------------
    dt = datetime.now()
    date_and_time = (dt.strftime("%x").replace("/", "-") + "_"
                     + dt.strftime("%H") + "h"
                     + dt.strftime("%M") + "m"
                     + dt.strftime("%M") + "s")  # MM-DD-YY_HHhMMmSSs
    del dt

    # -------------------------------- Parser --------------------------------
    parser = argparse.ArgumentParser()

    # Paths and options
    parser.add_argument("--data_path", type=str,
                        default="./data/preprocessed_MTPL.csv",
                        help="path to the data")
    parser.add_argument("--gan_path", type=str,
                        default="./saved_models/baseline_gan.dat",
                        help="path to the GAN (for the generator)")
    parser.add_argument("--ae_path", type=str,
                        default="./saved_models/baseline_ae.dat",
                        help="path to the autoencoder (for the decoder)")
    parser.add_argument("--random_seed", type=int, default=13,
                        help="fix random number generator sequence")
    parser.add_argument("--results_path", type=str,
                        default=Path.home() / f"mncdp/eval_{date_and_time}/",
                        help="path to folder where to save the results")
    parser.add_argument("--cpu_threads", type=int,
                        default=torch.get_num_threads(),
                        help="number of threads to run on if using CPU")
    parser.add_argument("--train_ratio", type=float, default=2/3,
                        help="ratio of the samples in the training set")

    # Save choices as parser
    args = parser.parse_args()

    # -------------------------------- Setup --------------------------------
    # Set random seeds
    seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

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
                                  + str(Path("/test.log"))),
                        format='%(message)s',
                        filemode="w",  # Overwrite instead of appending
                        level=logging.DEBUG)

    # Disable matplotlib debugging logger
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)

    # Log the parameters of the evaluation
    logging.info("Parameters:")
    for name, value in vars(args).items():
        logging.info(f"\t{name} = {value}")
    logging.info("\n")

    # Load pretrained generator
    with open(Path(args.gan_path), 'rb') as saved_generator:
        if device == "cpu":
            generator = torch.load(saved_generator, map_location="cpu")
        else:
            generator = torch.load(saved_generator)
    generator.to(device)

    # Load pretrained autoencoder and get the decoder
    with open(Path(args.ae_path), 'rb') as saved_autoencoder:
        if device == "cpu":
            autoencoder = torch.load(saved_autoencoder, map_location="cpu")
        else:
            autoencoder = torch.load(saved_autoencoder)
    decoder = autoencoder.get_decoder()
    decoder.to(device)

    # ----------------------------- Prepare data -----------------------------
    # Load and process data
    real_processed, datatypes, processor, real_df, names_of_cat = prepare_data(
        Path(args.data_path).absolute(),
        mtpl_features + autoencoder.feature_choice,
        is_eval=True,
    )

    # Log the type of each feature
    logging.info("Type of each feature:")
    for feature_name, feature_type in datatypes:
        logging.info(f"\t{feature_name} : {feature_type}")
    logging.info("\n")

    # Generate data in latent space and decode it (N x 30+)
    synthetic_decoded = decoder(
        generator(
            torch.randn(len(real_processed), generator.latent_dim).to(device)
        )
    ).to("cpu").detach().numpy()

    # Apply post-processing to obtain same format as real data (N x 9)
    synthetic = processor.inverse_transform(synthetic_decoded)  # numeric only

    # Round the floating point features to integer when needed (N x 30+)
    synthetic_decoded = processor.transform(synthetic)

    # Put the generated data in a DataFrame for easier plotting (N x 9)
    synthetic_df = pd.DataFrame(synthetic, columns=real_df.columns)

    # Log model
    logging.info(generator)
    logging.info("Decoder")  # The name is not included in the decoder
    logging.info(decoder)
    logging.info("\n")

    # ----------------------- Univariate distributions -----------------------
    print("Plotting and saving the univariate distributions...")

    for feature_name in real_df.columns:
        print(f"\t{feature_name}")
        # Set name of the file to save the plot
        save_name = (str(args.results_path)
                     + str(Path(f"/distribution_{feature_name}.png")))
        # Get the categories labels for this feature
        cat_labels = names_of_cat[feature_name]

        # If there's too many categories, barplots don't look good
        if len(cat_labels) > 30:
            # So we plot a relative count histogram instead
            plot_numeric(real_df[feature_name],
                         synthetic_df[feature_name],
                         feature_name,
                         save_name
                         )
        else:
            # Plot relative counts as barplot
            plot_categorical(real_df[feature_name],
                             synthetic_df[feature_name],
                             feature_name,
                             cat_labels,
                             save_name
                             )

    print("... Univariate distributions saved.\n")

    # ---------------------- RF predictions on ClaimNb ----------------------
    # Note that Random Forest models don't require normalization of features

    # Index where to separate train and test sets
    train_cutoff = round(args.train_ratio * len(real_df))

    # Predictions on ClaimNb, which is assumed to be the last feature
    print("Training and testing the RF models...")
    print("\ton ClaimNb")

    # Find the exact name of the "ClaimNb column (the target)
    target_name = real_df.columns[-1]

    # Separate features from target
    x_real = real_processed[:, :-1]  # If ClaimNb (only one column)
    y_real = np.array(real_df[target_name])  # target is the value of the cat
    x_syn = synthetic_decoded[:, :-1]
    y_syn = synthetic[:, -1]
    if target_name == "ClaimNb_cat":  # Remove the one-hot of ClaimNb_cat
        n_one_hot = int(real_df[target_name].nunique())  # One vector per cat
        x_real = real_processed[:, :-n_one_hot]
        x_syn = synthetic_decoded[:, :-n_one_hot]

    # Split train/test
    x_real_train = x_real[:train_cutoff]
    y_real_train = y_real[:train_cutoff]
    x_syn_train = x_syn[:train_cutoff]
    y_syn_train = y_syn[:train_cutoff]

    # Test set is always from the real data
    x_test = x_real[train_cutoff:]
    y_test = y_real[train_cutoff:]

    # Get the name of the feature for each of the processed column
    names_of_features = get_processed_feature_names(datatypes[:-1], real_df)

    logging.info("---------- RF Predictions on ClaimNb ----------\n")

    # Regression
    logging.info("-------- Regression results --------\n")
    print("\t\tRegression")

    # When trained on the real data
    logging.info("-- Trained on the real data --\n")
    results_msg = rf_regression(x_real_train, y_real_train,
                                x_test, y_test,
                                names_of_features)
    logging.info(results_msg)

    # When trained on the synthetic data
    logging.info("-- Trained on the generated data --\n")
    results_msg = rf_regression(x_syn_train, y_syn_train,
                                x_test, y_test,
                                names_of_features)
    logging.info(results_msg)
    
    # Multiclass Classification
    logging.info("-------- Classification results --------\n")
    print("\t\tClassification")

    # When trained on the real data
    logging.info("-- Trained on the real data --\n")
    save_name = (str(args.results_path) + str(Path("/cm_real.png")))
    results_msg = rf_classification(x_real_train, y_real_train,
                                    x_test, y_test,
                                    names_of_features,
                                    save_name,
                                    cm_as_plot=True)
    logging.info(results_msg)

    # When trained on the synthetic data
    logging.info("-- Trained on the generated data --\n")
    save_name = (str(args.results_path) + str(Path("/cm_generated.png")))
    results_msg = rf_classification(x_syn_train, y_syn_train,
                                    x_test, y_test,
                                    names_of_features,
                                    save_name,
                                    cm_as_plot=True)
    logging.info(results_msg)

    # ---------------------- Poisson GLM results ----------------------
    # TODO: Implement this

    print(f"\nResults were saved in the folder: {args.results_path}")
