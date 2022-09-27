"""Output functions."""
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pyart  # Need to load pyart even though not used to access colormaps
import torch
from cmcrameri import cm as cm_crameri


QTY_FORMATS = {
    "DBZH": "{x:.0f}",
    "VRAD": "{x:.0f}",
    "SNR": "{x:.0f}",
    "ZDR": "{x:.1f}",
    "RHOHV": "{x:.2f}",
    "KDP": "{x:.1f}",
    "HCLASS": "{x:1.0f}",
    "PHIDP": "{x:.2f}",
    "SQI": "{x:.1f}",
    "TH": "{x:.0f}",
    "WRAD": "{x:.1f}",
    "LOG": "{x:.0f}",
    "RR": "{x:.0f}",
    "RR_diff": "{x:.0f}",
}


QTY_RANGES = {
    "DBZH": (-15.0, 60.0),
    "HCLASS": (1.0, 6.0),
    "KDP": (-4.0, 8.0),
    "PHIDP": (0, 360.0),
    "RHOHV": (0.8, 1.0),
    "SQI": (0.0, 1.0),
    "TH": (-15.0, 60.0),
    "VRAD": (-30.0, 30.0),
    "WRAD": (0.0, 5.0),
    "ZDR": (-4.0, 4.0),
    "SNR": (-30.0, 50.0),
    "LOG": (0.0, 50.0),
    "RR": (1.0, 10.0),
    "RR_diff": (-5.0, 5.0),
}

COLORBAR_TITLES = {
    "DBZH": "Equivalent reflectivity factor (dBZ)",
    "HCLASS": "HydroClass",
    "KDP": "Specific differential phase (degrees/km)",
    "PHIDP": "Differential phase (degrees)",
    "RHOHV": "Copolar correlation coefficient",
    "SQI": "Normalized coherent power",
    "TH": "Total reflectivity factor (dBZ)",
    "VRAD": "Radial velocity (m/s)",
    "WRAD": "Doppler spectrum width (m/s)",
    "ZDR": "Differential reflectivity (dB)",
    "SNR": "Signal-to-noise ratio (dB)",
    "LOG": "LOG signal-to-noise ratio (dB)",
    "RR": "Rain rate [mm h$^{-1}$]",
    "RR_diff": "Rain rate [mm h$^{-1}$]",
}


def plot_1h_plus_1h_timeseries(model, dataset, indices=[10, 529]):
    """Plot a dBZ timeseries of model output.

    The timeseries is created as:
    - 1h of observations
    - 1h of nowcasts
    - 1h of observations corresponding to the nowcasts.

    Parameters
    ----------
    model : pytorch/pytorch-lightning model
        The model that is used to create nowcasts.
    dataset : datasets.fmi.FMIComposite dataset
        The dataset from which the data picked.
    indices : list, optional
        Indices that are picked from the dataset, by default [10, 529]

    """
    for idx in indices:
        arr = dataset[idx]
        window_ = dataset.get_window(idx)
        to_dbz = dataset.from_grayscale
        arr_1 = dataset[idx - 7]
        window_1 = dataset.get_window(idx - 7)
        # Plot observations
        for i in range(5):
            plot_dbz_image(
                to_dbz(arr_1[0][i, ...]).detach().cpu().numpy().squeeze(),
                window_1[i],
                "observation",
            )
        for i in range(7):
            plot_dbz_image(
                to_dbz(arr_1[1][i, ...]).detach().cpu().numpy().squeeze(),
                window_1[5 + i],
                "observation",
            )
        # Plot target
        window_2 = dataset.get_window(idx + 5)
        arr_2 = dataset[idx + 5]
        for i in range(5):
            plot_dbz_image(
                to_dbz(arr_2[0][i, ...]).detach().cpu().numpy().squeeze(),
                window_2[i],
                "target",
            )
        for i in range(7):
            plot_dbz_image(
                to_dbz(arr_2[1][i, ...]).detach().cpu().numpy().squeeze(),
                window_2[5 + i],
                "target",
            )
        # Plot forecast
        output = model(torch.unsqueeze(arr[0], 0).float(), future_steps=15)
        for i in range(12):
            plot_dbz_image(
                to_dbz(output[0, i, ...]).detach().cpu().numpy().squeeze(),
                window_[5 + i],
                "nowcast",
            )


def plot_array(ax, arr, colorbar=True, qty="DBZH", cmap=None, norm=None, extend="max"):
    """Plot an array with pcolormesh to axis.

    Parameters
    ----------
    ax : matplotlib axis
        The axis that the image is plotted on.
    dbz_arr : np.ndarray
        The dBZ image.
    colorbar : bool
        Whether to plot colorbar.
    qty : str
        Quantity name to select colormap, norm and colorbar label
    norm : matplotlib norm or None
        Norm for the image. If none, default values for quantity are applied.
    cmap : matplotlib cmap or None
        Colormap for the image. If none, default values for quantity are applied.
    """
    cbar_ax_kws = {
        "width": "5%",  # width = 5% of parent_bbox width
        "height": "100%",
        "loc": "lower left",
        "bbox_to_anchor": (1.01, 0.0, 1, 1),
        "borderpad": 0,
    }

    if cmap is None or norm is None:
        cmap, norm = _get_colormap(qty)

    # Mask invalid dBZ values
    arr = np.ma.masked_where(arr < -20, arr)
    # Final image is flipped
    ax.pcolormesh(
        np.flipud(arr),
        cmap=cmap,
        norm=norm,
        zorder=10,
        rasterized=True,
    )

    cbar = None
    # Add colorbar
    if colorbar:
        cax = inset_axes(ax, bbox_transform=ax.transAxes, **cbar_ax_kws)
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            format=mpl.ticker.StrMethodFormatter(QTY_FORMATS[qty]),
            orientation="vertical",
            cax=cax,
            ax=None,
            pad=0.1,
            extend=extend,
            ticks=np.arange(
                QTY_RANGES[qty][0],
                QTY_RANGES[qty][1] + 0.1,
                1,
            ),
        )
        # cbar.locator = mpl.ticker.MultipleLocator(2)
        # cbar.update_ticks()
        cbar.ax.tick_params(labelsize="small")
        cbar.set_label(label=COLORBAR_TITLES[qty], weight="normal")

    # No ticklabels
    ax.set_aspect(1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return cbar


def plot_dbz_image(dbz_arr, day, datatype, figsize=(12, 10)):
    """Plot a dBZ image.

    Parameters
    ----------
    dbz_arr : np.ndarray
        The dBZ image.
    day : datetime.datetime
        Time of the image, output in the image title and filename.
    datatype : str
        Extension that is added to the output filename, e.g. 'nowcast'.
    figsize : tuple of (width, height)
        The figure size.

    """
    cbar_ax_kws = {
        "width": "5%",  # width = 5% of parent_bbox width
        "height": "100%",
        "loc": "lower left",
        "bbox_to_anchor": (1.01, 0.0, 1, 1),
        "borderpad": 0,
    }
    qty = "DBZH"
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=figsize, sharex="col", sharey="row"
    )

    cax = inset_axes(ax, bbox_transform=ax.transAxes, **cbar_ax_kws)

    cmap, norm = _get_colormap(qty)

    # Mask invalid dBZ values
    dbz_arr = np.ma.masked_where(dbz_arr < -20, dbz_arr)
    # Final image is flipped
    ax.pcolormesh(
        np.flipud(dbz_arr),
        cmap=cmap,
        norm=norm,
        zorder=10,
    )

    # Add colorbar
    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        format=mpl.ticker.StrMethodFormatter(QTY_FORMATS[qty]),
        orientation="vertical",
        cax=cax,
        ax=None,
    )
    cbar.set_label(label=COLORBAR_TITLES[qty], weight="bold")
    cbar.ax.tick_params(labelsize=12)

    # No ticklabels
    ax.set_aspect(1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.suptitle(f"{day:%Y/%m/%d %H:%M} UTC {datatype}", y=0.92)
    fig.savefig(f"dbz_{day:%Y%m%d%H%M}_{datatype}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def debug_training_plot(
    epoch, batch, time_list, inputs, outputs, targets, quantity="DBZH", write_fig=True
):
    fig, axarr = plt.subplots(
        3, outputs.shape[1], figsize=(outputs.shape[1] * 4, 15), squeeze=False
    )
    cmap, norm = _get_colormap(quantity)

    for t in range(outputs.shape[1]):

        try:
            axarr[0][t].imshow(
                inputs[0, t, ...].detach().cpu().numpy(),
                cmap=cmap,
                norm=norm,
                origin="lower",
            )
        except IndexError:
            axarr[0][t].axis("off")

        axarr[1][t].imshow(
            targets[0, t, ...].detach().cpu().numpy(),
            cmap=cmap,
            norm=norm,
            origin="lower",
        )
        axarr[2][t].imshow(
            outputs[0, t, ...].detach().cpu().numpy(),
            cmap=cmap,
            norm=norm,
            origin="lower",
        )

        if time_list is not None:
            axarr[0][t].set_title(f"Input {t}: {time_list[t]}")
            axarr[1][t].set_title(f"Target {t}: {time_list[t+inputs.shape[1]]}")
            axarr[2][t].set_title(f"Prediction {t}: {time_list[t+inputs.shape[1]]}")
        else:
            axarr[0][t].set_title(f"Input {t}")
            axarr[1][t].set_title(f"Target {t}")
            axarr[2][t].set_title(f"Prediction {t}")

    fig.subplots_adjust(wspace=0.2, hspace=0.01)
    if write_fig:
        plt.savefig(f"{epoch:03d}_{batch:03d}.png", bbox_inches="tight")
    plt.close()
    return fig


def _get_colormap(quantity):
    if quantity == "HCLASS":
        cmap = colors.ListedColormap(["r", "b", "g", "y", "k", "c"])
        norm = colors.BoundaryNorm(np.arange(0.5, 7.5), cmap.N)
    elif "VRAD" in quantity:
        cmap = "pyart_BuDRd18"
        norm = None
    elif "DBZH" in quantity:
        cmap = "pyart_NWSRef"
        norm = None
    elif quantity == "TH":
        cmap = "pyart_NWSRef"
        norm = None
    elif "SNR" in quantity or "LOG" in quantity:
        cmap = "pyart_Carbone17"
        norm = None
    elif quantity == "KDP":
        cmap = "pyart_Theodore16"
        norm = None
    elif quantity == "PHIDP":
        cmap = "pyart_Wild25"
        norm = None
    elif quantity == "RHOHV":
        bounds = [0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 0.94, 0.96, 0.98, 0.99, 1.05]
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        cmap = cm.get_cmap("pyart_RefDiff", len(bounds))
    elif "WRAD" in quantity:
        cmap = "pyart_NWS_SPW"
        norm = None
    elif quantity == "ZDR":
        cmap = "pyart_RefDiff"
        norm = None
    elif quantity == "RR":
        cmap = "viridis"
        bounds = np.arange(
            QTY_RANGES[quantity][0],
            QTY_RANGES[quantity][1] + 0.1,
            0.5,
        )
        cmap = plt.get_cmap(cmap, len(bounds))
        norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        # norm = colors.LogNorm(
        #     vmin=QTY_RANGES[quantity][0], vmax=QTY_RANGES[quantity][1]
        # )
    elif quantity == "RR_diff":
        cmap = "cmc.vik"
        bounds = np.arange(
            QTY_RANGES[quantity][0],
            QTY_RANGES[quantity][1] + 0.1,
            0.5,
        )
        cmap = plt.get_cmap(cmap, len(bounds))
        norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        # norm = colors.LogNorm(
        #     vmin=QTY_RANGES[quantity][0], vmax=QTY_RANGES[quantity][1]
        # )
    else:
        cmap = cm.jet
        norm = None

    if norm is None:
        norm = colors.Normalize(
            vmin=QTY_RANGES[quantity][0], vmax=QTY_RANGES[quantity][1]
        )
    return cmap, norm
