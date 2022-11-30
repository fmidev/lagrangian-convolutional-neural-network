"""Plot example nowcasts.

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""
import argparse
import pyart

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import os
from datetime import datetime

from pathlib import Path

import contextily as ctx

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

    outdir = Path(conf.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # how many nowcasts to plot
    nrows = 2 + len(conf.nowcasts.keys())
    ncols = max(len(conf.leadtimes), conf.n_input_images)

    # Map parameters
    if conf.plot_map:
        _ = ctx.bounds2raster(
            *conf.map_params.bbox_lonlat,
            ll=True,
            path="map.tif",
            zoom=conf.map_params.zoom,
            source=ctx.providers.Stamen.TonerLite,
        )
        map_im = plt.imread("map.tif")

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=conf.figsize,
        sharex="row",
        sharey="col",
    )

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
        nan_mask = np.isnan(obs[i])

        cbar = plot_array(
            axes[0, i], obs[i], qty="RR", colorbar=(i == conf.n_input_images - 1)
        )
        axes[0, i].set_title(times[i][:-3])

        axes[0, i].pcolormesh(
            np.zeros_like(obs[0]),
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

        # Plot advection field
        if adv_field is not None:
            axes[0, i].quiver(
                adv_field_x[::quiver_thin, ::quiver_thin],
                np.flipud(adv_field_y)[::quiver_thin, ::quiver_thin],
                adv_field[0, ...][::quiver_thin, ::quiver_thin],
                -1 * np.flipud(adv_field[1, ...])[::quiver_thin, ::quiver_thin],
                linewidth=adv_field_lw,
                color=adv_field_color,
                alpha=adv_field_alpha,
            )

        if conf.plot_map:
            axes[0, i].imshow(
                map_im, zorder=0, extent=[0, obs[0].shape[0], 0, obs[0].shape[1]]
            )

    cbar.ax.yaxis.label.set_size("x-small")
    cbar.ax.tick_params(labelsize="x-small")

    for i in range(conf.n_input_images, ncols):
        axes[0, i].set_axis_off()

    axes[0, 0].set_ylabel("Input")

    # Plot target
    for i in range(len(conf.leadtimes)):
        obs[conf.n_input_images + i][
            obs[conf.n_input_images + i] < conf.min_val
        ] = np.nan
        plot_array(axes[1, i], obs[conf.n_input_images + i], qty="RR", colorbar=False)
        axes[1, i].set_title(f"{date:%Y-%m-%d %H:%M} + {conf.leadtimes[i] * 5:>3} min ")

        axes[1, i].pcolormesh(
            np.zeros_like(obs[0]),
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

        # Plot advection field
        if adv_field is not None:
            axes[1, i].quiver(
                adv_field_x[::quiver_thin, ::quiver_thin],
                np.flipud(adv_field_y)[::quiver_thin, ::quiver_thin],
                adv_field[0, ...][::quiver_thin, ::quiver_thin],
                -1 * np.flipud(adv_field[1, ...])[::quiver_thin, ::quiver_thin],
                linewidth=adv_field_lw,
                color=adv_field_color,
                alpha=adv_field_alpha,
            )
        if conf.plot_map:
            axes[1, i].imshow(
                map_im, zorder=0, extent=[0, obs[0].shape[0], 0, obs[0].shape[1]]
            )

    axes[1, 0].set_ylabel("Target")

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

            if conf.plot_diff:
                arr = nowcasts[method][i] - obs[conf.n_input_images + i]
                plot_array(
                    axes[j + 2, i],
                    arr,
                    qty="RR_diff",
                    colorbar=(i == ncols - 1),
                    extend="both",
                )
            else:
                nowcasts[method][i][nowcasts[method][i] < conf.min_val] = np.nan
                plot_array(
                    axes[j + 2, i], nowcasts[method][i], qty="RR", colorbar=False
                )
            axes[j + 2, i].pcolormesh(
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
            # axes[j + 2, i].set_title(times[conf.n_input_images + i])
            axes[j + 2, i].set_title(
                f"{date:%Y-%m-%d %H:%M} + {conf.leadtimes[i] * 5:>3} min "
            )

            # Plot advection field
            if adv_field is not None:
                axes[j + 2, i].quiver(
                    adv_field_x[::quiver_thin, ::quiver_thin],
                    np.flipud(adv_field_y)[::quiver_thin, ::quiver_thin],
                    adv_field[0, ...][::quiver_thin, ::quiver_thin],
                    -1 * np.flipud(adv_field[1, ...])[::quiver_thin, ::quiver_thin],
                    linewidth=adv_field_lw,
                    color=adv_field_color,
                    alpha=adv_field_alpha,
                )
            if conf.plot_map:
                axes[j + 2, i].imshow(
                    map_im, zorder=0, extent=[0, obs[0].shape[0], 0, obs[0].shape[1]]
                )

        axes[j + 2, 0].set_ylabel(conf.nowcasts[method]["title"])

    if conf.plot_map:
        COPYRIGHT_TEXT = "Map tiles by Stamen Design, under CC BY 3.0. Map data by OpenStreetMap, under ODbL."
        fig.text(0.99, -0.005, COPYRIGHT_TEXT, fontsize=4, zorder=10, ha="right")

    for ax in axes.flat:
        ax.set_xticks(np.linspace(0, obs[0].shape[0], 5))
        ax.set_yticks(np.linspace(0, obs[0].shape[1], 5))

        ax.grid(lw=0.5, color="tab:gray", ls=":", zorder=11)

        for tick in ax.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        for tick in ax.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(True)

    # fig.subplots_adjust()
    fig.savefig(
        outdir / date.strftime(conf.filename),
        bbox_inches="tight",
        dpi=conf.dpi,
    )
