import numpy as np
from pysteps import verification

from pincast_verif.metrics import Metric
from pincast_verif.plot_tools import plot_1d


class CategoricalMetric(Metric):
    def __init__(
        self, leadtimes, cat_metrics, thresh, tables: dict = None, **kwargs
    ) -> None:

        self.name_template = "CAT_t_{thresh}_l_{leadtime}"
        self.leadtimes = leadtimes
        self.thresholds = thresh
        self.cat_metrics = cat_metrics
        if tables is None:
            self.tables = {}
            for lt in leadtimes:
                for thr in thresh:
                    name = self.name_template.format(leadtime=lt, thresh=thr)
                    self.tables[name] = verification.det_cat_fct_init(thr=thr)
            self.is_empty = True
        else:
            self.tables = tables
            self.is_empty = False

    def accumulate(self, x_pred, x_obs):
        if x_pred.ndim == 4:
            x_pred = x_pred.mean(axis=1)
        for i, lt in enumerate(self.leadtimes):
            for thr in self.thresholds:
                name = self.name_template.format(leadtime=lt, thresh=thr)
                verification.det_cat_fct_accum(
                    contab=self.tables[name], pred=x_pred[i], obs=x_obs[i]
                )
            self.is_empty = False

    def compute(self):
        names = []
        values = []
        for metric in self.cat_metrics:
            for thr in self.thresholds:
                thr_str = str(thr).replace(".", "_")
                out_name = "{metric}_{thr}".format(metric=metric, thr=thr_str)
                names.append(out_name)
                metric_values = np.empty(len(self.leadtimes))
                for i, lt in enumerate(self.leadtimes):
                    in_name = self.name_template.format(thresh=thr, leadtime=lt)
                    metric_values[i] = verification.det_cat_fct_compute(
                        self.tables[in_name], scores=metric
                    )[metric]
                values.append(metric_values)
        return np.array(values), names

    def merge(self, categorical_other):
        self.tables = {
            name: verification.det_cat_fct_merge(table, categorical_other.tables[name])
            for name, table in self.tables.items()
        }

    @staticmethod
    def plot(
        scores: dict,
        method: str,
        lt: np.array,
        exp_id: str,
        path_save: str,
        method_plot_params: dict = {},
        subplot_kwargs: dict = {},
        plot_kwargs: dict = {},
        cat_kwargs : dict = {}
    ):
        return plot_1d(
            scores,
            method,
            lt,
            exp_id,
            path_save,
            subplot_kwargs=subplot_kwargs,
            plot_kwargs=plot_kwargs,
            method_plot_params=method_plot_params,
            **cat_kwargs
        )
