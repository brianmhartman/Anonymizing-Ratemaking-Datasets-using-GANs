#!/usr/bin/env python
# coding=utf-8

"""
Script to preprocess the MTPL data for the auto-dpgan
"""

import argparse

import pandas as pd
import numpy as np


if __name__ == "__main__":
    # -------------------------------- Parser --------------------------------
    parser = argparse.ArgumentParser(
        description="Simple preprocessing of MTPL data. The resulting"
                    "csv contains all relevant features in the right type."
                    "For the features that can be considered either numeric or"
                    "categorical, both versions are present. For example, "
                    "'Density' is numeric while 'Density_cat' is categoric."
    )
    parser.add_argument("--data_path", type=str,
                        default="./data/OriginalMTPLData.csv",
                        help="path to the MTPL data")

    args = parser.parse_args()

    # ---------------------------- Preprocessing ----------------------------
    df = pd.read_csv(args.data_path)
    df = df.drop(columns="PolicyID")  # Remove useless PolicyID index
    df = df[df.Exposure <= 1]  # Remove samples with Exposure > 1

    # Add the region, exposure and density as categories

    # Load the csv file with the preprocessing already done
    cat_df = pd.read_csv("./data/policy_dat_v2.csv")
    cat_df = cat_df[["Region", "Exposure_cat", "Density_cat"]]

    # Update the df
    df.Region = cat_df.Region  # Replace region names with codes
    df["Exposure_cat"] = cat_df.Exposure_cat.astype("category")
    df["Density_cat"] = cat_df.Density_cat.astype("category")
    del cat_df  # Clear memory

    # Binning preprocessing

    # Exposure: 13 bins (1 / month + 1 for the value 1.0)
    cut_bins = list(np.linspace(0, 1, 12, endpoint=False))  # 1 bin / month
    cut_bins.extend([1 - 1e-16, 1])  # Add bin for the value 1 only
    df["Exposure_bin"] = pd.cut(df.Exposure, bins=cut_bins)
    exposure_map = {}  # Map bins to integer values for readability
    for index, value in enumerate(sorted(df["Exposure_bin"].unique())):
        exposure_map[value] = index + 1  # categories 1 to 13
    df["Exposure_bin"] = df["Exposure_bin"].map(exposure_map)

    # CarAge: 11 bins (10 quantiles + 1 for the value 0.0)
    cut_bins = [-1, 0, 1, 2, 3, 4, 6, 8, 10, 12, 15, int(max(df.CarAge))]
    df["CarAge_bin"] = pd.cut(df.CarAge, bins=cut_bins)

    # DriverAge: 10 bins (sizes of bins increase with age)
    cut_bins = [int(min(df.DriverAge)) - 1, 20, 23, 26, 30,
                35, 45, 55, 65, 75, int(max(df.DriverAge))]
    df["DriverAge_bin"] = pd.cut(df.DriverAge, bins=cut_bins)

    # Density: 10 bins (quantiles) of log(Density)
    log_density = pd.Series(np.log(df.Density))
    df["Log(Density)_bin"] = pd.qcut(log_density, q=10)

    # Last details

    # Change categorical features type (requires less storage space)
    for col_name, col_series in df.iteritems():
        if col_series.dtypes == object:  # If values are strings
            df[col_name] = col_series.astype("category")

    # Add ClaimNb categorical version
    df["ClaimNb_cat"] = df.ClaimNb.astype("category")

    # Reorder columns (changing the order will break already trained models)
    features_in_order = ["Power", "Brand", "Gas", "Region",
                         "DriverAge", "DriverAge_bin",
                         "CarAge", "CarAge_bin",
                         "Density", "Density_cat", "Log(Density)_bin",
                         "Exposure", "Exposure_cat", "Exposure_bin",
                         "ClaimNb", "ClaimNb_cat"]
    df = df.reindex(columns=features_in_order)

    # Save preprocessed dataframe as csv
    df.to_csv("./data/preprocessed_MTPL.csv", index=False)
