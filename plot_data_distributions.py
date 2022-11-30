"""Plot distributions of data"""
# import os
import sys
from pathlib import Path
import argparse
import random
import numpy as np

import seaborn as sns
import zarr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import dask.array as da
import dask

from utils.config import load_config
from utils.logging import setup_logging

import torch
from torch.utils.data import DataLoader

from datasets import LagrangianFMIComposite


TITLES = {
    "train": "a) Training data set",
    "valid": "b) Validation data set",
    "test": "c) Test data set",
}


def main(configpath, splits):
    confpath = Path("config") / configpath
    dsconf = load_config(confpath / "lagrangian_datasets.yaml")

    plt.style.use("distributions.mplstyle")

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    rng = (0, 100)
    n_bins = 100
    batch_size = 15

    hists = {}

    for split in splits:

        dset = LagrangianFMIComposite(split=split, **dsconf.fmi)
        dloader = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=batch_size,
            shuffle=True,
            pin_memory=False,
        )

        hist = None
        bins = None
        count = 0

        n = 0
        for batch in dloader:
            inp, _, _ = batch
            h_, bins = np.histogram(
                inp[:, -1, :, :].numpy().ravel(), bins=n_bins, range=rng
            )

            count += sum(h_)

            if hist is None:
                hist = h_
            else:
                hist += h_

            n += 1

        hists[split] = {
            "hist": hist,
            "count": count,
        }

    # Plot histogram
    _, bins = np.histogram([], bins=n_bins, range=rng)

    nrows = len(splits)
    ncols = 1
    fig, axes = plt.subplots(
        figsize=(3.5, nrows * 2.1),
        nrows=nrows,
        ncols=ncols,
        squeeze=True,
        sharex="row",
        sharey="row",
    )

    width = bins[-1] - bins[-2]
    for i, split in enumerate(splits):
        axes[i].bar(
            bins[:-1],
            hists[split]["hist"],
            width=width,
            align="edge",
            color="k",
            edgecolor="k",
            zorder=10,
        )
        axes[i].set_title(TITLES[split])

    for ax in axes.flat:

        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
        ax.set_ylim(bottom=0)
        ax.set_xlim(rng)
        ax.grid(which="major", lw=0.5, color="tab:gray", ls="-", zorder=0)
        ax.grid(which="minor", lw=0.5, color="tab:gray", ls="-", alpha=0.1, zorder=0)

        ax.set_yscale("log")

        ax.set_ylabel("Count")
        ax.set_xlabel("Rain rate [mm h$^{-1}$]")

    outpath = Path(args.outpath)
    outpath.parents[0].mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("config", type=str, help="Configuration folder")
    argparser.add_argument(
        "splits", type=str, nargs="+", help="Dataset splits that are processed"
    )
    argparser.add_argument("outpath", type=str, help="Output file path")
    args = argparser.parse_args()

    main(args.config, args.splits)
