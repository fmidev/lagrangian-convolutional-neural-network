"""
Script reading saved metrics for chosen models/ nowcasting methods from npy file
and plotting average metric values versus leadtime.
"""
import argparse
import os
from attrdict import AttrDict
import yaml
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pincast_verif.io_tools import read_file
from pincast_verif.plot_tools import get_done_df_stats, plot_data_quality
from pincast_verif.metrics import *


def run(config, config_path=None):

    plt.ioff()

    # CONFIG
    config_copy_path = config.config_copy_path
    metric_exp_ids = config.metric_exp_ids
    done_csv_path = config.done_csv_path
    metrics_path_npy = config.metrics_npy_path
    name_path = config.name_path
    metrics = config.metrics
    methods = config.methods

    exp_id = config.exp_id
    path_save = config.path_save

    leadtimes = np.array(config.leadtimes) * config.timestep
    dq_plot_params = config.dq_plot
    cont_plot_params = config.cont_plot
    cat_plot_params = config.cat_plot
    fss_plot_params = config.fss_plot
    rapsd_plot_params = config.rapsd_plot
    is_plot_params = config.intensity_scale_plot
    ssim_plot_params = config.ssim_plot

    if config.stylefile is not None:
        plt.style.use(config.stylefile)

    # MAKE DIRS, COPY CONFIG FILE
    for m in list(methods.keys()) + ["ALL", "DQ"]:
        os.makedirs(
            os.path.dirname(path_save.format(id=exp_id, method=m, metric="")),
            exist_ok=True,
        )
    shutil.copyfile(src=config_path, dst=config_copy_path.format(id=exp_id))

    # DATA QUALITY
    done_dfs = {id: pd.read_csv(done_csv_path.format(id=id)) for id in metric_exp_ids}
    df_stats = []
    for i, df in done_dfs.items():
        df_stats += get_done_df_stats(df, i)
    plot_data_quality(df_stats, exp_id=exp_id, path_save=path_save, **dq_plot_params)

    # FETCH SCORES
    scores = dict()
    for metric in metrics:
        scores[metric] = {}
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
                scores[metric].update({method: (npy_now, name_now)})

        # Write to pd dataframe and csv file
        methods_dfs = {}
        for method_name, score_tuple in scores[metric].items():
            methods_dfs[method_name] = pd.DataFrame(
                score_tuple[0],
                columns=np.arange(1, score_tuple[0].shape[1] + 1),
                index=score_tuple[1],
            )
        score_df = pd.concat(methods_dfs.values(), keys=methods_dfs.keys())
        csv_fn = (
            Path(path_save.format(id=exp_id, method="ALL", metric=metric)).parents[1]
            / f"{metric}_values.csv"
        )

        # Sort to get nicer excels
        score_df.index = score_df.index.swaplevel(0, 1)
        score_df.sort_index(inplace=True)
        score_df.to_csv(csv_fn)

        # 1) compare all models for one metric at a time, save each as a figure
        if metric == "CAT":
            CategoricalMetric.plot(
                scores=scores[metric],
                method="ALL",
                lt=leadtimes,
                exp_id=exp_id,
                path_save=path_save,
                method_plot_params=methods,
                cat_kwargs=cat_plot_params,
            )
        if metric == "CONT":
            ContinuousMetric.plot(
                scores=scores[metric],
                method="ALL",
                lt=leadtimes,
                exp_id=exp_id,
                path_save=path_save,
                method_plot_params=methods,
                cont_kwargs=cont_plot_params,
            )
        if metric == "FSS":
            FssMetric.plot(
                scores=scores[metric],
                method="ALL",
                lt=leadtimes,
                exp_id=exp_id,
                path_save=path_save,
                method_plot_params=methods,
                fss_kwargs=fss_plot_params,
            )
        if metric == "RAPSD":
            RapsdMetric.plot(
                data=scores[metric],
                method="ALL",
                leadtimes=rapsd_plot_params.lts,
                exp_id=exp_id,
                path_save=path_save,
                method_plot_params=methods,
                kwargs=rapsd_plot_params,
            )
        if metric == "CRPS":
            Crps.plot(
                scores=scores[metric],
                method="ALL",
                lt=leadtimes,
                exp_id=exp_id,
                path_save=path_save,
                method_plot_params=methods,
                crps_kwargs=cont_plot_params,
            )
        if metric == "SSIM":
            SSIMMetric.plot(
                scores=scores[metric],
                method="ALL",
                lt=leadtimes,
                exp_id=exp_id,
                path_save=path_save,
                method_plot_params=methods,
                ssim_kwargs=ssim_plot_params,
            )
        # 2) again for one model at a time
        for method in methods.keys():
            if metric == "CAT":
                CategoricalMetric.plot(
                    scores=scores[metric],
                    method=method,
                    lt=leadtimes,
                    exp_id=exp_id,
                    path_save=path_save,
                    method_plot_params=methods,
                    cat_kwargs=cat_plot_params,
                )
            if metric == "CONT":
                ContinuousMetric.plot(
                    scores=scores[metric],
                    method=method,
                    lt=leadtimes,
                    exp_id=exp_id,
                    path_save=path_save,
                    method_plot_params=methods,
                    cont_kwargs=cont_plot_params,
                )
            if metric == "FSS":
                FssMetric.plot(
                    scores=scores[metric],
                    method=method,
                    lt=leadtimes,
                    exp_id=exp_id,
                    path_save=path_save,
                    method_plot_params=methods,
                    fss_kwargs=fss_plot_params,
                )
            if metric == "RAPSD":
                RapsdMetric.plot(
                    data=scores[metric],
                    method=method,
                    leadtimes=rapsd_plot_params.lts,
                    exp_id=exp_id,
                    path_save=path_save,
                    method_plot_params=methods,
                    kwargs=rapsd_plot_params,
                )
            if metric == "INTENSITY_SCALE":
                IntensityScaleMetric.plot(
                    exp_id=exp_id,
                    scores=scores[metric],
                    method=method,
                    path_save_fmt=path_save,
                    thresh=is_plot_params.thresh,
                    scales=is_plot_params.scales,
                    method_plot_params=methods,
                    kmperpixel=is_plot_params.kmperpixel,
                    vminmax=is_plot_params.vminmax,
                )
            if metric == "CRPS":
                Crps.plot(
                    scores=scores[metric],
                    method=method,
                    lt=leadtimes,
                    exp_id=exp_id,
                    path_save=path_save,
                    method_plot_params=methods,
                    crps_kwargs=cont_plot_params,
                )
            if metric == "SSIM":
                SSIMMetric.plot(
                    scores=scores[metric],
                    method=method,
                    lt=leadtimes,
                    exp_id=exp_id,
                    path_save=path_save,
                    method_plot_params=methods,
                    ssim_kwargs=ssim_plot_params,
                )
        del scores[metric]


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("config_path", type=str, help="Configuration file path")
    args = argparser.parse_args()

    with open(args.config_path, "r") as f:
        config = AttrDict(yaml.safe_load(f))

    run(config, config_path=args.config_path)
