"""
Utility functions for working with metrics
"""
from pincast_verif.metrics import *
from functools import reduce


def get_metric(
    metric_name: str, leadtimes: list, metric_params: dict, tables: dict = None
):
    """Match metric names to objects, containing
    1. A contingency table under the "table" attribute.
    2. An "accumulate(x_pred, x_obs)" method.
    3. A "compute() method.

    To add a new metric, create it under the metrics folder and add
    the matching choice here.

    Args:
        metric_name (str): _description_
        leadtimes (list): _description_
        metric_params (dict): _description_

    Returns:
        _type_: _description_
    """
    if metric_name.upper() == "RAPSD":
        return RapsdMetric(tables=tables, **metric_params.rapsd)
    elif metric_name.upper() == "CONT":
        return ContinuousMetric(leadtimes, tables=tables, **metric_params)
    elif metric_name.upper() == "CAT":
        return CategoricalMetric(leadtimes, tables=tables, **metric_params)
    elif metric_name.upper() == "FSS":
        return FssMetric(leadtimes, tables=tables, **metric_params)
    elif metric_name.upper() == "INTENSITY_SCALE":
        return IntensityScaleMetric(leadtimes, tables=tables, **metric_params)
    elif metric_name.upper() == "CRPS":
        return Crps(leadtimes=leadtimes, tables=tables, **metric_params)
    elif metric_name.upper() == "SSIM":
        return SSIMMetric(leadtimes, tables=tables, **metric_params.ssim)
    elif metric_name.upper() == "RANK_HISTOGRAM":
        return RankHistogram(**metric_params.rank_histogram, tables=tables)
    elif metric_name.upper() == "PROB":
        return ProbabilisticMetric(
            thresholds = metric_params.thresholds,
            **metric_params.prob,
            tables=tables
            )
    else:
        raise NotImplementedError(f"Metric {metric_name.upper()} not implemented.")


def merge_metrics(metric_instance_1, metric_instance_2):
    if (not hasattr(metric_instance_1, "merge")) or (
        not hasattr(metric_instance_2, "merge")
    ):
        raise AttributeError(
            f"Either {metric_instance_1} or {metric_instance_2} does not have a 'merge' attribute."
        )
    if (not hasattr(metric_instance_1, "is_empty")) or (
        not hasattr(metric_instance_2, "is_empty")
    ):
        raise AttributeError(
            f"Either {metric_instance_1} or {metric_instance_2} does not have a 'is_empty' attribute."
        )
    if not type(metric_instance_1) is type(metric_instance_2):
        raise TypeError(
            f"{metric_instance_1} and {metric_instance_2} are not of the same type."
        )

    if metric_instance_1.is_empty and metric_instance_2.is_empty:
        return metric_instance_1
    elif metric_instance_1.is_empty:
        return metric_instance_2
    elif metric_instance_2.is_empty:
        return metric_instance_1
    else:
        metric_instance_1.merge(metric_instance_2)
        return metric_instance_1


def merge_outdict(a, b, path=None):
    "merges b into a"
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_outdict(a[key], b[key], path + [str(key)])
            else:
                a[key] = merge_metrics(
                    metric_instance_1=a[key], metric_instance_2=b[key]
                )
        else:
            raise Exception("Outdicts do not match at %s" % ".".join(path + [str(key)]))
    return a


def merge_outdict_list(outdict_list: list):
    return reduce(merge_outdict, outdict_list)


def merge_boolean_df_list(done_df_list: list):
    return reduce(lambda ddf1, ddf2: ddf1 | ddf2, done_df_list)
