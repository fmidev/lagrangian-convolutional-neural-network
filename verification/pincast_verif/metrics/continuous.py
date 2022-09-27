import numpy as np
from pysteps import verification

from pincast_verif.metrics import Metric
from pincast_verif.plot_tools import plot_1d


class ContinuousMetric(Metric):
    def __init__(self, leadtimes, cont_metrics, tables: dict = None, **kwargs) -> None:

        self.name_template = "CONT_l_{leadtime}"
        self.leadtimes = leadtimes
        self.cont_metrics = cont_metrics
        if tables is None:
            self.tables = {}
            for lt in leadtimes:
                name = self.name_template.format(leadtime=lt)
                self.tables[name] = verification.det_cont_fct_init()
            self.is_empty = True
        else:
            self.tables = tables
            self.is_empty = False

    def accumulate(self, x_pred, x_obs):
        if x_pred.ndim == 4:
            x_pred = x_pred.mean(axis=1)
        for i, lt in enumerate(self.leadtimes):
            name = self.name_template.format(leadtime=lt)
            verification.det_cont_fct_accum(
                err=self.tables[name], pred=x_pred[i], obs=x_obs[i]
            )
        self.is_empty = False

    def compute(self):
        names = []
        values = []
        for metric in self.cont_metrics:
            out_name = "{metric}".format(metric=metric)
            names.append(out_name)
            metric_values = np.empty(len(self.leadtimes))
            for i, lt in enumerate(self.leadtimes):
                in_name = self.name_template.format(leadtime=lt)
                metric_values[i] = verification.det_cont_fct_compute(
                    self.tables[in_name], scores=metric
                )[metric]
            values.append(metric_values)
        return np.array(values), names

    def merge(self, continuous_other):
        self.tables = {
            name: verification.det_cont_fct_merge(table, continuous_other.tables[name])
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
        cont_kwargs: dict = {},
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
            **cont_kwargs
        )
