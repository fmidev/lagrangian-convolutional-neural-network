"""Output functions."""
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pyart  # Need to load pyart even though not used to access colormaps
from cmcrameri import cm as cm_crameri


QTY_FORMATS = {
    "DBZH": "{x:.0f}",
    "VRAD": "{x:.0f}",
    "SNR": "{x:.0f}",
    "ZDR": "{x:<4.1f}",
    "RHOHV": "{x:.2f}",
    "KDP": "{x:.1f}",
    "HCLASS": "{x:1.0f}",
    "PHIDP": "{x:.2f}",
    "SQI": "{x:.1f}",
    "TH": "{x:.0f}",
    "WRAD": "{x:.1f}",
    "LOG": "{x:.0f}",
    "R": "{x:<4.0f}",
    "R_pysteps": "{x:.1f}",
    "R_log": "{x:.1f}",
    "RR_diff": "{x:.0f}",
    "H": lambda x, p: f"{x/1000:<4.1f}",
    "H_ml": lambda x, p: f"{x/1000:<4.1f}",
    "FZT": "{x:.0f}",
    "TRENDSS": "{x:<4.1f}",
    "zdr_column": lambda x, p: f"{x/1000:<4.1f}",
}


QTY_RANGES = {
    "DBZH": (-15.0, 60.0),
    "HCLASS": (1.0, 6.0),
    "KDP": (-2.5, 7.5),
    "PHIDP": (0, 360.0),
    "RHOHV": (0.8, 1.0),
    "SQI": (0.0, 1.0),
    "TH": (-15.0, 60.0),
    "VRAD": (-30.0, 30.0),
    "WRAD": (0.0, 5.0),
    "ZDR": (-4.0, 5.0),
    "SNR": (-30.0, 50.0),
    "LOG": (0.0, 50.0),
    "R": (0.1, 10.0),
    "R_pysteps": (0.1, 100.0),
    "R_log": (0.1, 50.0),
    "RR_diff": (-10.0, 10.0),
    "H": (0.0, 20000.0),
    "H_ml": (0.0, 5000.0),
    "FZT": (1000.0, 4000.0),
    "TRENDSS": (0.0, 12.0),
    "zdr_column": (600.0, 4000.0),
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
    "R": "Rain rate [mm h$^{-1}$]",
    "R_pysteps": "Rain rate [mm h$^{-1}$]",
    "R_log": "Rain rate [mm h$^{-1}$]",
    "RR_diff": "Rain rate [mm h$^{-1}$]",
    "TRENDSS": "Normalized ZDR anomaly",
    "H": "Height [km]",
    "H_ml": "Height [km]",
    "zdr_column": "ZDR column height [km]",
    "FZT": "Freezing level [m]",
}


def plot_array(
    ax,
    arr,
    x=None,
    y=None,
    colorbar=True,
    qty="DBZH",
    cmap=None,
    norm=None,
    extend="max",
    flip=False,
    zorder=10,
):
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
        cmap, norm, ticks = _get_colormap(qty)

    if extend != "min" and extend != "both":
        # Set values outside colorbar range to nan
        arr[arr < norm.vmin] = np.nan
    if extend != "max" and extend != "both":
        # Set values outside colorbar range to nan
        arr[arr > norm.vmax] = np.nan

    # Mask invalid dBZ values
    # arr = np.ma.masked_where(arr < -20, arr)
    # Final image is flipped
    if flip:
        arr = np.flipud(arr)

    if x is not None and y is not None:
        xx, yy = np.meshgrid(x, y)
        args = (xx, yy, arr)
    else:
        args = (arr,)

    ax.pcolormesh(
        *args,
        cmap=cmap,
        norm=norm,
        zorder=zorder,
        rasterized=True,
    )

    cbar = None
    # Add colorbar
    if colorbar:
        cax = inset_axes(ax, bbox_transform=ax.transAxes, **cbar_ax_kws)

        if callable(QTY_FORMATS[qty]):
            formatter = mpl.ticker.FuncFormatter(QTY_FORMATS[qty])
        else:
            formatter = mpl.ticker.StrMethodFormatter(QTY_FORMATS[qty])

        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            format=formatter,
            orientation="vertical",
            cax=cax,
            ax=None,
            pad=0.1,
            extend=extend,
            ticks=ticks,
            # ticks=np.arange(
            #     QTY_RANGES[qty][0],
            #     QTY_RANGES[qty][1] + 0.1,
            #     1,
            # ),
        )
        # cbar.locator = mpl.ticker.MultipleLocator(2)
        # cbar.update_ticks()
        if isinstance(norm, mpl.colors.LogNorm):
            cbar.ax.set_yscale("log")
        if isinstance(norm, mpl.colors.TwoSlopeNorm):
            cbar.ax.set_yscale("linear")
        if ticks is not None:
            cbar.set_ticks(ticks)
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


def plot_input_pred_multifeature_array(
    input, pred, conf, input_times=None, pred_times=None, **kwargs
):
    """Plot input-prediction figure for multiple features.

    This function is used to plot the input and prediction
    for multiple features in the LogNowcast callback.

    Parameters
    ----------
    input : np.ndarray
        The input data, assumed to have dimensions (n_features, n_input_times, H, W).
    pred : np.ndarray
        The prediction data, assumed to have dimensions
        (n_features, n_pred_times, H, W).
    conf : dict
        Configuration dictionary for each feature, assumed to have structure:
        { int : { cmap_qty: str, title: str }}
        where the first key int is the index of the feature in the data.
    input_times : list[datetime.datetime], optional
        Input times. If given, written in subplot title. By default None
    pred_times : _type_, optional
        Prediction times. If given, written in subplot title, by default None

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.

    """
    nrows = input.shape[0]
    ncols = max(input.shape[1], pred.shape[1])

    fig = plt.figure(figsize=(ncols * 4 + 1, nrows * 8), constrained_layout=True)
    subfigs = fig.subfigures(nrows=nrows, ncols=1, squeeze=False)

    for irow in range(nrows):
        # Plot each feature
        subfig = subfigs[irow, 0]
        axs = subfig.subplots(nrows=2, ncols=ncols, squeeze=True)
        subfig.supylabel(conf[irow]["title"], fontsize="large")

        # Inputs on first row
        for it in range(input.shape[1]):
            plot_array(
                axs[0, it],
                input[irow, it],
                qty=conf[irow]["cmap_qty"],
                colorbar=(it == (input.shape[1] - 1)),
                extend=conf[irow]["extend"],
            )
            if input_times is not None:
                axs[0, it].set_title(input_times[it].strftime("%Y-%m-%d %H:%M"))

        # Predictions on second row
        for it in range(pred.shape[1]):
            if pred.shape[0] <= irow:
                # No prediction for this feature
                break
            plot_array(
                axs[1, it],
                pred[irow, it],
                qty=conf[irow]["cmap_qty"],
                colorbar=False,
                extend=conf[irow]["extend"],
            )
            if pred_times is not None:
                axs[1, it].set_title(pred_times[it].strftime("%Y-%m-%d %H:%M"))

        for ax in axs.flat:
            ax.set_aspect(1)

            # Remove ticks and labels
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

        # Clear axis where we didn't plot anything
        for it in range(input.shape[1], ncols):
            axs[0, it].axis("off")
        for it in range(pred.shape[1], ncols):
            axs[1, it].axis("off")

    return fig


def _get_colormap(quantity):
    ticks = None
    if quantity == "HCLASS":
        cmap = mpl.colors.ListedColormap(
            ["peru", "dodgerblue", "blue", "cyan", "yellow", "red"]
        )
        norm = mpl.colors.BoundaryNorm(np.arange(0.5, 7.5), cmap.N)
    elif "VRAD" in quantity:
        cmap = "cmc.roma_r"
        norm = None
    elif "DBZH" in quantity:
        bounds = np.arange(QTY_RANGES[quantity][0], QTY_RANGES[quantity][1] + 1, 5)
        norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        cmap = cm.get_cmap("pyart_HomeyerRainbow", len(bounds))
    elif quantity == "TH":
        bounds = np.arange(QTY_RANGES[quantity][0], QTY_RANGES[quantity][1] + 1, 5)
        norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        cmap = cm.get_cmap("pyart_HomeyerRainbow", len(bounds))
    elif "SNR" in quantity or "LOG" in quantity:
        cmap = "pyart_Carbone17"
        norm = None
    elif quantity == "KDP":
        bounds = np.arange(
            QTY_RANGES[quantity][0], QTY_RANGES[quantity][1] + 0.01, 0.25
        )
        norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        cmap = cm.get_cmap("pyart_Theodore16", len(bounds))
        ticks = np.arange(
            np.ceil(QTY_RANGES[quantity][0]), QTY_RANGES[quantity][1] + 0.01, 1.0
        )
    elif quantity == "PHIDP":
        cmap = "pyart_Wild25"
        norm = None
    elif quantity == "RHOHV":
        bounds = [0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 0.94, 0.96, 0.98, 0.99, 1.05]
        norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        cmap = cm.get_cmap("pyart_RefDiff", len(bounds))
        ticks = bounds
    elif "WRAD" in quantity:
        cmap = "pyart_NWS_SPW"
        norm = None
    elif quantity == "ZDR":
        bounds = np.arange(QTY_RANGES[quantity][0], QTY_RANGES[quantity][1] + 0.1, 0.5)
        norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        cmap = cm.get_cmap("pyart_LangRainbow12", len(bounds))
    elif quantity == "R":
        cmap = "viridis"
        bounds = np.arange(
            QTY_RANGES[quantity][0],
            QTY_RANGES[quantity][1] + 0.5,
            1.0,
        )
        cmap = plt.get_cmap(cmap, len(bounds))
        norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        # norm = mpl.colors.LogNorm(
        #     vmin=QTY_RANGES[quantity][0], vmax=QTY_RANGES[quantity][1]
        # )
    elif quantity == "R_pysteps":
        cmap = "viridis"
        bounds = [
            0.1,
            0.2,
            0.5,
            1.5,
            2.5,
            4,
            6,
            10,
            15,
            20,
            30,
            40,
            50,
            60,
            75,
            100,
            150,
        ]
        colors = [
            "cyan",
            "deepskyblue",
            "dodgerblue",
            "blue",
            "chartreuse",
            "limegreen",
            "green",
            "darkgreen",
            "yellow",
            "gold",
            "orange",
            "red",
            "magenta",
            "darkmagenta",
        ]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "cmap", colors, len(bounds) - 1
        )
        cmap.set_over("darkred", 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        ticks = bounds
    elif quantity == "R_log":
        cmap = "viridis"
        bounds = [0.1, 0.2, 0.5, 1, 1.5, 2.5, 5, 7.5, 10, 15, 25, 40, 65, 100]
        cmap = plt.get_cmap(cmap, len(bounds))
        norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        # norm = mpl.colors.LogNorm(
        #     vmin=QTY_RANGES[quantity][0], vmax=QTY_RANGES[quantity][1]
        # )
        ticks = bounds
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
    elif quantity == "TRENDSS":
        cmap_upper = "cmc.hawaii"
        cmap_lower = "cmc.grayC"

        norm = mpl.colors.TwoSlopeNorm(
            vmin=QTY_RANGES[quantity][0], vcenter=3, vmax=QTY_RANGES[quantity][1]
        )

        colors_lower = plt.cm.get_cmap(cmap_lower)(np.linspace(0, 1.0, 100))
        colors_upper = plt.cm.get_cmap(cmap_upper)(np.linspace(0, 1.0, 100))

        # Set alpha
        colors_lower[:, -1] = 0.6

        all_colors = np.vstack((colors_lower, colors_upper))
        cmap = mpl.colors.LinearSegmentedColormap.from_list("trendss_cmap", all_colors)

        ticks = [
            QTY_RANGES[quantity][0],
            *np.arange(3, QTY_RANGES[quantity][1] + 0.01, 1.0),
        ]

    else:
        cmap = "cmc.hawaii_r"
        norm = None

    if norm is None:
        norm = mpl.colors.Normalize(
            vmin=QTY_RANGES[quantity][0], vmax=QTY_RANGES[quantity][1]
        )
    return cmap, norm, ticks