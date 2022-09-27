"""Functions for plotting verification scores."""
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator
import seaborn as sns
import numpy as np


FSS_USEFUL_LABEL = "1st leadtime where \nFSS < FSS$_\\mathrm{useful}$"
FSS_USEFUL_PLOT_KWS = {
    "marker": "o",
    "mfc": "white",
    "ms": 3,
    "linestyle": 'None',
    "zorder": 11,
}
FSS_USEFUL_PATCH = mpl.lines.Line2D(
    [], [], mec="k", **FSS_USEFUL_PLOT_KWS, label=FSS_USEFUL_LABEL)


def plot_cont_scores_against_leadtime(
        df, outfn="pysteps_scores.png", max_leadtime=80, write_fig=True):
    """Plot continuous scores.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe storing the scores. Should have column "Leadtime" denoting
        the forecast leadtime.
    outfn : str
        Output filename. Possible preceding directories should exist.
    max_leadtime : int
        Maximum plotted leadtime.
    write_fig : bool
        Whether to write figure.

    Returns
    -------
    matplotlib.pyplot.Figure

    """
    stats = df.columns.values.tolist()

    leadtimes = np.unique(df["Leadtime"])

    try:
        stats.remove("Leadtime")
    except (IndexError, ValueError):
        pass

    # Create figure
    ncols = 1
    nrows = 1
    fig, axes = plt.subplots(
        figsize=(3 * ncols, 2.8 * nrows), nrows=nrows, ncols=ncols,
        constrained_layout=True,
        sharex=True, sharey=True,
        squeeze=False)
    # customPalette = ['#a6cee3', '#1f78b4', '#b2df8a']

    dfs = df.melt(
        id_vars=["Leadtime"], value_vars=stats,
        var_name='Score', value_name="Value"
    )
    # Ensure value is float to prevent hashing issues
    # (caused issues when dealing w/dataframes directly from calculations)
    dfs["Value"] = dfs["Value"].astype(float)
    sns.lineplot(
        data=dfs,
        x="Leadtime",
        y="Value",
        hue="Score",
        # hue_order=stats,
        # style=type_name, style_order=type_names,
        # palette=customPalette,
        ax=axes[0, 0],
    )
    ylims = [0, np.ceil(df[stats].max().max().item())]
    yres = 1.0
    for ax in axes.flat:
        _set_ax_background(ax)

    for ax in axes[:, 0]:
        _set_yaxis(ax, ylims, yres, " / ".join(stats))

    for ax in axes[-1, :]:
        _set_xaxis(ax, leadtimes, max_leadtime)

    if write_fig:
        fig.savefig(outfn, bbox_inches="tight", dpi=300)
    plt.close()
    return fig


def plot_cat_scores_against_leadtime(
        df, outfn="pysteps_scores.png", max_leadtime=80, write_fig=True):
    """Plot categorical scores.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe storing the scores. Should have column "Leadtime" denoting
        the forecast leadtime.
    outfn : str
        Output filename. Possible preceding directories should exist.
    max_leadtime : int
        Maximum plotted leadtime.
    write_fig : bool
        Whether to write figure.

    Returns
    -------
    matplotlib.pyplot.Figure

    """
    stats = df.columns.values.tolist()

    leadtimes = np.unique(df["Leadtime"])

    thresholds = np.unique(df["Threshold"])

    try:
        stats.remove("Leadtime")
        stats.remove("Threshold")
    except (IndexError, ValueError):
        pass

    # Create figure
    ncols = 1
    nrows = 1
    fig, axes = plt.subplots(
        figsize=(3 * ncols, 2.8 * nrows), nrows=nrows, ncols=ncols,
        constrained_layout=True,
        sharex=True, sharey=True, squeeze=False)
    # customPalette = ['#a6cee3', '#1f78b4', '#b2df8a']

    dfs = df.melt(
        id_vars=["Leadtime", "Threshold"], value_vars=stats,
        var_name='Score', value_name="Value"
    )
    # Ensure value is float to prevent hashing issues
    # (caused issues when dealing w/dataframes directly from calculations)
    dfs["Value"] = dfs["Value"].astype(float)

    # Plot
    sns.lineplot(
        data=dfs,
        x="Leadtime",
        y="Value",
        hue="Score",
        hue_order=stats,
        style="Threshold",
        style_order=thresholds,
        # palette=customPalette,
        ax=axes[0, 0],
    )

    ylims = [0, np.ceil(df[stats].max().max().item())]
    yres = 0.1
    for ax in axes.flat:
        _set_ax_background(ax)

    for ax in axes[:, 0]:
        _set_yaxis(ax, ylims, yres, " / ".join(stats))

    for ax in axes[-1, :]:
        _set_xaxis(ax, leadtimes, max_leadtime)

    if write_fig:
        fig.savefig(outfn, bbox_inches="tight", dpi=300)
    plt.close()
    return fig


def plot_fss_against_leadtime(
        df, max_leadtime=80, write_fig=True, grid_res=1000,
        outfn="fss.pdf"):
    """Plot FSS score comparison for multiple variables.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe storing the scores. Should have column "Leadtime" denoting
        the forecast leadtime, "FSS" for the FSS values, "Scale" for spatial scale,
        and "f0" for the FSS_useful value
    grid_res : float
        Grid resolution of the original grid used to calculate the scores, in meters.
    outfn : str
        Output filename. Possible preceding directories should exist.
    max_leadtime : int
        Maximum plotted leadtime.
    write_fig : bool
        Whether to write figure.
    max_leadtime : int
        Maximum plotted leadtime.

    """
    stat = "FSS"

    leadtimes = sorted([t for t in df["Leadtime"].unique() if t <= max_leadtime])

    # df_fss = df[["Leadtime"] + [c for c in df.columns if stat in c or "f0" in c]]
    df = df[df["Leadtime"].isin(leadtimes)]

    scales = df["Scale"].unique()
    thrs = df["Threshold"].unique()
    # Create figure
    fig = plt.figure(figsize=(5, 2.8), constrained_layout=True)

    # Create axis
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 0.2], height_ratios=[1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")

    # Get palette with correct amount of colors
    n_colors = len(scales)
    customPalette = sns.color_palette("mako", n_colors=n_colors)

    # Plot actual FSS curves
    df["FSS"] = df["FSS"].astype(float)
    sns.lineplot(
        data=df,
        x="Leadtime",
        y="FSS",
        hue="Scale",
        hue_order=scales,
        style="Threshold",
        style_order=thrs,
        palette=customPalette,
        ax=ax1,
        zorder=10,
    )

    # Plot FSS_useful values by marking the leadtime
    # when the fss value is below it
    _plot_FFS_useful(
        ax1, scales, df, leadtimes, customPalette)

    # Put legend outside of axis box
    handles, _ = ax1.get_legend_handles_labels()
    _remove_legend(ax1)
    _set_FSS_legend(handles, ax_leg)

    ylims = [0.2, 1.0]
    yres = 0.1
    # Setup axes
    for ax in [ax1]:
        _set_yaxis(ax, ylims, yres, stat)
        _set_xaxis(ax, leadtimes, max_leadtime)
        _set_ax_background(ax)

    # ax1.set_title(title)
    if write_fig:
        fig.savefig(outfn, bbox_inches="tight", dpi=300)
    plt.close()
    return fig


def _plot_FFS_useful(ax, scales, full_df, leadtimes, palette):
    # Plot FSS_useful values by marking the leadtime
    # when the fss value is below it
    for i, scale in enumerate(scales):
        f0_df = full_df[full_df["Scale"] == scale]
        # f0_df["FSS_useful"] = f0_df["FSS_useful"]
        color = palette[i]

        flag = False
        for lt in leadtimes:
            val = f0_df[f0_df["Leadtime"] == lt]["FSS"].values[0]
            fss_useful = f0_df[f0_df["Leadtime"] == lt]["FSS_useful"].values[0]

            if not flag and val < fss_useful:
                ax.plot(lt, val, mec=color, **FSS_USEFUL_PLOT_KWS)
                flag = True
            if flag:
                break


def _set_FSS_legend(handles, legend_ax):
    handles.append(FSS_USEFUL_PATCH)
    leg = legend_ax.legend(
        handles=handles,
        handlelength=1.2,
        loc=1,
        frameon=False,
        columnspacing=1
    )
    text = leg.get_texts()[-1]
    props = text.get_font_properties().copy()
    text.set_fontproperties(props)
    text.set_size(7)


def _remove_legend(ax):
    try:
        ax.get_legend().remove()
    except AttributeError:
        pass


def _set_yaxis(ax, ylims, yres, label):
    ax.set_ylabel(label)
    ax.set_ylim(ylims)
    ax.yaxis.set_major_locator(FixedLocator(
        locs=np.arange(ylims[0], ylims[1] + 0.01, yres)))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))


def _set_xaxis(ax, leadtimes, max_leadtime, xres=10):
    ax.set_xlim((2.5, max_leadtime + 2.5))
    ax.set_xlabel("Leadtime [min]")
    ax.xaxis.set_major_locator(FixedLocator(
        locs=np.arange(min(leadtimes), (max(leadtimes)) + 0.01, xres)))
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))


def _set_ax_background(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(which="both", linewidth=0.5)
