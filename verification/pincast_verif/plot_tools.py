"""
Utility functions for plotting verification metrics
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .io_tools import read_file


def get_all_names(metric_dict: dict):
    all_names = set()
    for _, names_i in metric_dict.values():
        all_names.update(names_i)
    return list(all_names)


def get_score_df(metrics, metric_exp_ids, name_path, metrics_path_npy, methods):
    scoredf = pd.DataFrame(columns=["type", "metric", "model", "lt", "value"])
    scoredf["value"] = scoredf["value"].astype(object)
    for metric in metrics:
        for method in methods.keys():
            for id in metric_exp_ids:
                try:
                    name_fn = name_path.format(id=id, metric=metric, method=method)
                    npy_fn = metrics_path_npy.format(
                        id=id, metric=metric, method=method
                    )
                    name_now = read_file(name_fn)
                    npy_now = np.load(npy_fn)
                except:
                    continue
                for i, n in enumerate(name_now):
                    for lt, val in enumerate(npy_now[i]):
                        s = pd.Series(
                            [metric, n, method, (lt + 1) * 5, val],
                            index=scoredf.columns,
                        )
                        scoredf = scoredf.append(s.T, ignore_index=True)
    return scoredf


def get_plot_name(name, unit="\mathrm{mm\,h}^{-1}", kmperpixel=1):

    parts = name.split("_")
    if len(parts) == 1:
        return name, name

    elif len(parts) == 3:
        # Format as [0] R_thr = [1].[2]
        return (
            parts[0]
            + " $R_\mathrm{thr} = "
            + parts[1]
            + "."
            + parts[2]
            + "~"
            + unit
            + "$",
            parts[0],
        )
    elif len(parts) == 4 and parts[0] == "FSS":
        return (
            parts[0]
            + " (scale = "
            + str(int(parts[1]) * kmperpixel)
            + " km)"
            + " $R_\mathrm{thr} = "
            + parts[2]
            + "."
            + parts[3]
            + "~"
            + unit
            + "$",
            parts[0],
        )
    elif parts[0] == "INTENSITY":
        return "FSS", "FSS"

    return name, "Verification score value"


def get_done_df_stats(done_df: pd.DataFrame, i: str):
    stats = []
    for col in done_df:
        col_mod = i + "/" + col
        if "unnamed" not in col.lower():
            perc = done_df[col].value_counts(normalize=True)
            stats.append([col_mod, 1 - perc[False]])
    return stats


def plot_1d(
    scores: dict,
    method: str,
    lt: np.array,
    exp_id: str,
    path_save: str,
    ylim_val: list = [-0.05, 1.05],
    yticks: list = [0.0, 1.1, 0.1],
    xlim_val: list = [0.0, 200.0],
    xticks: list = [0.0, 200.0, 30.0],
    axis_type: str = "linear",
    plot_epx_id: bool = True,
    subplot_kwargs: dict = {},
    plot_kwargs: dict = {},
    method_plot_params: dict = {},
):

    plot_lt_indices = (lt / 5 - 1).astype(int)
    if method == "ALL":
        names = get_all_names(metric_dict=scores)
        plot = {name: plt.subplots(**subplot_kwargs) for name in names}
        for method_i, (val, names) in scores.items():
            for i, n in enumerate(names):
                if not "label" in method_plot_params[method_i].keys():
                    method_plot_params[method_i]["label"] = method_i
                plot_func = match_axis_type(axis_type, plot[n][1])
                plot_func(
                    lt,
                    val[i, plot_lt_indices],
                    **method_plot_params[method_i],
                    **plot_kwargs,
                )
    else:
        val, names = scores[method]
        plot = {name: plt.subplots(**subplot_kwargs) for name in names}
        for i, n in enumerate(names):
            if not "label" in method_plot_params[method].keys():
                method_plot_params[method]["label"] = method
            plot_func = match_axis_type(axis_type, plot[n][1])
            plot_func(
                lt, val[i, plot_lt_indices], **method_plot_params[method], **plot_kwargs
            )

    for _, ax in plot.values():
        ax.legend()
        ax.grid()
        ax.set_xticks(np.arange(*xticks))
        ax.set_xlim(xlim_val)
        if ylim_val is not None:
            ax.set_yticks(np.arange(*yticks))
            ax.set_ylim(ylim_val)
        ax.grid(linestyle="--")

    for name, (fig, ax) in plot.items():
        path_save_now = path_save.format(id=exp_id, method=method, metric=name)

        metric_label, metric_name = get_plot_name(name)

        if plot_epx_id:
            if method == "ALL":
                ax.set_title(f"{exp_id}: {metric_label}")
            else:
                ax.set_title(
                    f"{exp_id}: {metric_label} for {method_plot_params[method]['label']}"
                )
        else:
            if method == "ALL":
                ax.set_title(f"{metric_label}")
            else:
                ax.set_title(
                    f"{metric_label} for {method_plot_params[method]['label']}"
                )
        ax.set_xlabel("Leadtime [min]")
        ax.set_ylabel(metric_name)
        fig.savefig(path_save_now, bbox_inches="tight", dpi=600)
        plt.close(fig)


def plot_rapsd(
    data: dict,
    method: str,
    scales: np.ndarray,
    leadtimes: list,
    exp_id: str,
    path_save: str,
    kwargs: dict,
    method_plot_params: dict,
):
    for i, lt in enumerate(leadtimes):
        fig, ax = plt.subplots(**kwargs.subplot_kwargs)
        if method == "ALL":
            for meth, (arr, name) in data.items():
                if not "label" in method_plot_params[meth].keys():
                    method_plot_params[meth]["label"] = meth
                ax.plot(scales, arr[i], **method_plot_params[meth])
        else:
            arr, name = data[method]
            if not "label" in method_plot_params[method].keys():
                method_plot_params[method]["label"] = meth
            ax.plot(scales, arr[i], **method_plot_params[method])
        ax.grid()
        ax.legend()
        ax.set_xlabel("scale [km]")

        fig.savefig(
            path_save.format(
                id=exp_id, method=method, metric=str(lt * 5) + "min_" + "RAPSD"
            ),
            bbox_inches="tight",
            dpi=600,
        )


def plot_intensityscale(
    values,
    name,
    fig,
    thresh,
    scales,
    vminmax=None,
    kmperpixel=None,
    unit=None,
    title=None,
):
    """
    adapted from pysteps' plot_intensityscale function
    takes in : 2d threshs x scales  ; name : for ex. is_lt_1, thresh and scale list, figure
    """

    ax = fig.gca()

    metric_label, metric_name = get_plot_name(name, unit=unit, kmperpixel=kmperpixel)

    vmin = vmax = None
    if vminmax is not None:
        vmin = np.min(vminmax)
        vmax = np.max(vminmax)

    cmap = sns.color_palette("cubehelix", as_cmap=True)
    im = ax.imshow(values, vmin=vmin, vmax=vmax, interpolation="nearest", cmap=cmap)
    cb = fig.colorbar(im)
    cb.set_label(metric_name)

    if unit is None:
        ax.set_xlabel("Intensity threshold")
    else:
        ax.set_xlabel("Intensity threshold [%s]" % unit)
    if kmperpixel is None:
        ax.set_ylabel("Spatial scale [pixels]")
    else:
        ax.set_ylabel("Spatial scale [km]")

    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels(thresh)
    ax.set_yticks(np.arange(values.shape[0]))
    if kmperpixel is None:
        scales = np.flip(np.array(scales))
    else:
        scales = np.flip(np.array(scales)) * kmperpixel
    ax.set_yticklabels(scales)
    ax.set_title(title)


def plot_data_quality(
    stats_list: list, exp_id: str, path_save: str, label_rot: int = 0
) -> None:
    plt.bar(range(len(stats_list)), [val[1] * 100 for val in stats_list])
    plt.xticks(range(len(stats_list)), [val[0] for val in stats_list])
    plt.xticks(rotation=label_rot)
    plt.ylabel("percentage (%) of full data")
    plt.ylim([0.00, 105.0])
    plt.tight_layout()
    plt.savefig(path_save.format(id=exp_id, method="DQ", metric="data_quality"))
    plt.close()


def match_axis_type(axis_type_name, parent_axis) -> callable:
    if axis_type_name == "linear":
        return parent_axis.plot
    elif axis_type_name == "semilogy":
        return parent_axis.semilogy
    elif axis_type_name == "semilogx":
        return parent_axis.semilogx
    else:
        raise ValueError(f"axis type {axis_type} undefined")
