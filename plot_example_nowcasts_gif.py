"""Plot example nowcasts in separate figures.

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""
import argparse
import pyart

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import os
from datetime import datetime
import imageio

from pathlib import Path

from utils import plot_array, load_config, read_advection_fields_from_h5
from verification.pincast_verif import io_tools


pyart.load_config(os.environ.get("PYART_CONFIG"))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("configpath", type=str, help="Configuration file path")
    argparser.add_argument("date", type=str, help="date to be plotted (YYYYmmddHHMM")
    args = argparser.parse_args()

    date = datetime.strptime(args.date, "%Y%m%d%H%M")
    sample = date.strftime("%Y-%m-%d %H:%M:%S")

    confpath = Path(args.configpath)
    conf = load_config(confpath)
    plt.style.use(conf.stylefile)

    outdir = Path(conf.outdir) / date.strftime("%Y%m%d%H%M")
    outdir.mkdir(parents=True, exist_ok=True)

    duration_per_frame = 0.5

    # how many nowcasts to plot
    nrows = 2 + len(conf.nowcasts.keys())
    ncols = max(len(conf.leadtimes), conf.n_input_images)

    fig, axes = plt.subplots(
        nrows=1, ncols=1, figsize=(5, 6), sharex="col", sharey="row"
    )
    input_file = "input_%Y%m%d%H%M.png"
    output_file = "target_%Y%m%d%H%M.png"
    nowcast_file = lambda m: f"nowcast_{m}_%Y%m%d%H%M.png"

    dbs = dict()
    dbs["measurements"] = conf.measurements.path

    # Get observations
    get_times = [*list(range(-conf.n_input_images + 1, 1)), *conf.leadtimes]

    obs = io_tools.load_observations(
        dbs["measurements"],
        sample,
        leadtimes=get_times,
    )
    times = io_tools._get_sample_names(sample, get_times)
    try:
        obs = io_tools.dBZ_list_to_rainrate(obs)
    except:
        raise ValueError("Some observation missing!")

    # Read advection field
    if conf.advection_field_path is not None:
        adv_path = datetime.strftime(date, conf.advection_field_path)
        adv_fields = read_advection_fields_from_h5(adv_path)

        bbox_x_slice = slice(conf.adv_field_bbox[0], conf.adv_field_bbox[1])
        bbox_y_slice = slice(conf.adv_field_bbox[2], conf.adv_field_bbox[3])

        # TODO implement picking correct field if multiple exist
        adv_field = adv_fields[next(iter(adv_fields))][:, bbox_x_slice, bbox_y_slice]
        quiver_thin = 20
        adv_field_x, adv_field_y = np.meshgrid(
            np.arange(0, adv_field.shape[1]), np.arange(0, adv_field.shape[2])
        )
        adv_field_alpha = 1
        adv_field_lw = 0.7
        adv_field_color = "k"
    else:
        adv_field = None

    # Plot input
    for i in range(conf.n_input_images):
        obs[i][obs[i] < conf.min_val] = np.nan

        cbar = plot_array(axes, obs[i], qty="RR", colorbar=True)

        # Plot advection field
        if adv_field is not None:
            axes.quiver(
                adv_field_x[::quiver_thin, ::quiver_thin],
                np.flipud(adv_field_y)[::quiver_thin, ::quiver_thin],
                adv_field[0, ...][::quiver_thin, ::quiver_thin],
                -1 * np.flipud(adv_field[1, ...])[::quiver_thin, ::quiver_thin],
                linewidth=adv_field_lw,
                color=adv_field_color,
                alpha=adv_field_alpha,
            )

        axes.set_title(times[i][:-3])

        axes.set_xticks(np.linspace(0, obs[0].shape[0], 5))
        axes.set_yticks(np.linspace(0, obs[0].shape[1], 5))

        axes.grid(lw=0.5, color="tab:gray", ls=":")

        for tick in axes.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        for tick in axes.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

        for spine in ["top", "right"]:
            axes.spines[spine].set_visible(True)

        if cbar is not None:
            cbar.ax.yaxis.label.set_size("x-small")

        # axes[0, 0].set_ylabel("Observation")

        fig.savefig(
            outdir
            / datetime.strptime(times[i], "%Y-%m-%d %H:%M:%S").strftime(input_file),
            bbox_inches="tight",
            dpi=conf.dpi,
        )
        axes.clear()

    # build gif
    with imageio.get_writer(
        outdir / f"input_{date:%Y%m%d%H%M}.gif",
        format="GIF",
        mode="I",
        duration=duration_per_frame,
    ) as writer:
        for filename in sorted(outdir.glob("input_*.png")):
            image = imageio.imread(filename)
            writer.append_data(image)

    # Plot target
    for i in range(len(conf.leadtimes)):
        obs[conf.n_input_images + i][
            obs[conf.n_input_images + i] < conf.min_val
        ] = np.nan
        cbar = plot_array(axes, obs[conf.n_input_images + i], qty="RR", colorbar=True)
        axes.set_title(times[conf.n_input_images + i][:-3])

        # Plot advection field
        if adv_field is not None:
            axes.quiver(
                adv_field_x[::quiver_thin, ::quiver_thin],
                np.flipud(adv_field_y)[::quiver_thin, ::quiver_thin],
                adv_field[0, ...][::quiver_thin, ::quiver_thin],
                -1 * np.flipud(adv_field[1, ...])[::quiver_thin, ::quiver_thin],
                linewidth=adv_field_lw,
                color=adv_field_color,
                alpha=adv_field_alpha,
            )

        axes.set_xticks(np.linspace(0, obs[0].shape[0], 5))
        axes.set_yticks(np.linspace(0, obs[0].shape[1], 5))

        axes.grid(lw=0.5, color="tab:gray", ls=":")

        for tick in axes.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        for tick in axes.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

        for spine in ["top", "right"]:
            axes.spines[spine].set_visible(True)

        # axes.set_ylabel("Target")
        if cbar is not None:
            cbar.ax.yaxis.label.set_size("x-small")

        fig.savefig(
            outdir
            / datetime.strptime(
                times[conf.n_input_images + i], "%Y-%m-%d %H:%M:%S"
            ).strftime(output_file),
            bbox_inches="tight",
            dpi=conf.dpi,
        )
        axes.clear()

    # build gif
    with imageio.get_writer(
        outdir / f"target_{date:%Y%m%d%H%M}.gif",
        format="GIF",
        mode="I",
        duration=duration_per_frame,
    ) as writer:
        for filename in sorted(outdir.glob("target_*.png")):
            image = imageio.imread(filename)
            writer.append_data(image)

    # Load nowcasts
    nowcasts = io_tools.load_predictions(conf.nowcasts, sample, conf.leadtimes)

    if isinstance(nowcasts, str):
        raise ValueError(f"Some nowcast for {sample} for {nowcasts} missing!")

    # Plot nowcasts
    row = 2
    for j, method in enumerate(conf.nowcasts.keys()):
        try:
            nowcasts[method] = io_tools.dBZ_list_to_rainrate(nowcasts[method])
        except:
            raise ValueError(f"Some nowcast for {method} missing!")

        for i in range(len(conf.leadtimes)):
            nan_mask = np.isnan(nowcasts[method][i])
            nowcasts[method][i][nowcasts[method][i] < conf.min_val] = np.nan

            cbar = plot_array(axes, nowcasts[method][i], qty="RR", colorbar=True)
            axes.pcolormesh(
                np.flipud(nan_mask),
                cmap=colors.ListedColormap(
                    [
                        "white",
                        "tab:gray",
                    ]
                ),
                zorder=9,
                rasterized=True,
                vmin=0,
                vmax=1,
                alpha=0.5,
            )
            # axes.set_title(times[conf.n_input_images + i])
            axes.set_title(f"{date:%Y-%m-%d %H:%M} + {conf.leadtimes[i] * 5:>3} min ")

            # Plot advection field
            if adv_field is not None:
                axes.quiver(
                    adv_field_x[::quiver_thin, ::quiver_thin],
                    np.flipud(adv_field_y)[::quiver_thin, ::quiver_thin],
                    adv_field[0, ...][::quiver_thin, ::quiver_thin],
                    -1 * np.flipud(adv_field[1, ...])[::quiver_thin, ::quiver_thin],
                    linewidth=adv_field_lw,
                    color=adv_field_color,
                    alpha=adv_field_alpha,
                    zorder=11,
                )

            axes.set_xticks(np.linspace(0, obs[0].shape[0], 5))
            axes.set_yticks(np.linspace(0, obs[0].shape[1], 5))

            axes.grid(lw=0.5, color="tab:gray", ls=":", zorder=11)

            for tick in axes.xaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
            for tick in axes.yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

            for spine in ["top", "right"]:
                axes.spines[spine].set_visible(True)

            if cbar is not None:
                cbar.ax.yaxis.label.set_size("x-small")

            # axes[j + 2, 0].set_ylabel(conf.nowcasts[method]["title"])

            # fig.subplots_adjust()
            fig.savefig(
                outdir
                / datetime.strptime(
                    times[conf.n_input_images + i], "%Y-%m-%d %H:%M:%S"
                ).strftime(nowcast_file(method)),
                bbox_inches="tight",
                dpi=conf.dpi,
            )
            axes.clear()

        # build gif
        with imageio.get_writer(
            outdir / f"{method}_{date:%Y%m%d%H%M}.gif",
            format="GIF",
            mode="I",
            duration=duration_per_frame,
        ) as writer:
            for filename in sorted(outdir.glob(f"nowcast_{method}_*.png")):
                image = imageio.imread(filename)
                writer.append_data(image)
