#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Caleb Robinson <calebrob6@gmail.com>
#
"""Script for calculating per channel means and stdevs from a list of COGs
"""
import sys
import os

env = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    AWS_NO_SIGN_REQUEST="YES",
    GDAL_MAX_RAW_BLOCK_CACHE_SIZE="200000000",
    GDAL_SWATH_SIZE="200000000",
    VSI_CURL_CACHE_SIZE="200000000",
)
os.environ.update(env)
import time

import argparse
import numpy as np
import rasterio
import pandas as pd
import glob


def stats(verbose, input_fn, output_dir, num_samples_per_file, num_files, nodata):

    # -----------------------------------
    with open(input_fn, "r") as f:
        fns = f.read().strip().split("\n")
    if verbose:
        print("Found %d files" % (len(fns)))

    if num_files is not None:
        assert num_files <= len(
            fns
        ), "If you are going to sub-sample from the filelist, then you must specify a number of files less than the total number of files."
        np.random.shuffle(fns)
        fns = fns[:num_files]
        if verbose:
            print("...but only using %d of them" % (len(fns)))

    # -----------------------------------
    sampled_pixels = []

    if verbose:
        print("Sampling %d pixels per tile" % (num_samples_per_file))

    with rasterio.open(fns[0]) as f:
        num_channels = f.count

    tic = time.time()
    for i, fn in enumerate(fns):
        if i % 10 == 0 and verbose:
            print("%d/%d\t%0.2f seconds" % (i + 1, len(fns), time.time() - tic))
            tic = time.time()

        with rasterio.open(fn) as f:
            data = f.read().reshape(num_channels, -1)

        mask = np.sum(data == nodata, axis=0) == num_channels
        data = data[:, ~mask]
        num_samples = min(num_samples_per_file, data.shape[1])
        idxs = np.random.choice(data.shape[1], size=num_samples)

        pixels = data[:, idxs]
        sampled_pixels.append(pixels)

    sampled_pixels = np.concatenate(sampled_pixels, axis=1)
    means = sampled_pixels.mean(axis=1, dtype=np.float64)
    stdevs = sampled_pixels.std(axis=1, dtype=np.float64)

    # -----------------------------------

    print(type(means))
    print(type(means[0]))
    if output_dir is not None:
        # with open(args.output_fn, "w") as f:
        #     f.write("%s\n" % (means))
        #     f.write("%s\n" % (stdevs))
        df = pd.DataFrame.from_dict({"means": means, "stdevs": stdevs})
        df.to_csv(
            output_dir + "/" + os.path.splitext(os.path.basename(input_fn))[0] + ".csv"
        )

        df2 = pd.DataFrame.from_dict(
            {
                "name": os.path.splitext(os.path.basename(input_fn))[0],
                "means": [means],
                "stdevs": [stdevs],
            }
        )
        df2.to_csv(
            output_dir
            + "/"
            + os.path.splitext(os.path.basename(input_fn))[0]
            + "_2.csv"
        )

    means = ",".join(["%0.4f" % (val) for val in means])
    stdevs = ",".join(["%0.4f" % (val) for val in stdevs])

    if verbose:
        print("Means:", means)
        print("Stdevs:", stdevs)

    elif not verbose:
        print(means)
        print(stdevs)


def main():
    parser = argparse.ArgumentParser(description="Image statistic calculation script")

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debugging",
        default=False,
    )
    parser.add_argument(
        "--input_dir",
        action="store",
        type=str,
        help="Path to filelist. Filenames should be readable by rasterio.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        type=str,
        help="Filename to write (if this is not set, then we print the results to stdout)",
        default=None,
    )
    parser.add_argument(
        "--num_samples_per_file",
        action="store",
        type=int,
        help="Filename to write",
        default=10000,
    )
    parser.add_argument(
        "--num_files",
        action="store",
        type=int,
        help="Number of files to subsample",
        default=None,
    )
    parser.add_argument(
        "--nodata",
        action="store",
        type=int,
        help="The nodata value to check (we assume that if each band in the data equals this value, then the position is nodata)",
        default=0,
    )

    args = parser.parse_args(sys.argv[1:])

    f_lst = [
        args.input_dir + x for x in os.listdir(args.input_dir) if not x.startswith(".")
    ]
    print("f_list : ", f_lst)
    print("number of files to process: ", len(f_lst))

    for i, f in enumerate(f_lst):
        print(i)
        stats(
            args.verbose,
            f,
            args.output_dir,
            args.num_samples_per_file,
            args.num_files,
            args.nodata,
        )

    # Combine all dataframe together into master dataframe
    df_all = pd.concat(
        map(pd.read_csv, glob.glob(os.path.join(args.output_dir, "*_2.csv")))
    )
    df_all.to_csv(args.output_dir + "/" "all_stats.csv")


if __name__ == "__main__":
    main()
