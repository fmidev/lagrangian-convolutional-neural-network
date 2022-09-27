import numpy as np
from pysteps import verification

from pincast_verif.metrics import Metric
from pincast_verif.plot_tools import plot_1d


class FssMetric(Metric):
    def __init__(
        self, leadtimes, thresh, scales, tables: dict = None, **kwargs
    ) -> None:

        self.name_template = "FSS_s_{scale}_t_{thresh}_l_{leadtime}"
        self.leadtimes = leadtimes
        self.thresholds = thresh
        self.scales = scales
        if tables is None:
            self.tables = {}
            for lt in leadtimes:
                for thr in thresh:
                    for scale in scales:
                        name = self.name_template.format(
                            leadtime=lt, thresh=thr, scale=scale
                        )
                        self.tables[name] = verification.spatialscores.fss_init(
                            thr=thr, scale=scale
                        )
            self.is_empty = True
        else:
            self.tables = tables
            self.is_empty = False

    def accumulate(self, x_pred, x_obs):
        if x_pred.ndim == 4:
            x_pred = x_pred.mean(axis=1)
        for i, lt in enumerate(self.leadtimes):
            for thr in self.thresholds:
                for scale in self.scales:
                    name = self.name_template.format(
                        leadtime=lt, thresh=thr, scale=scale
                    )
                    verification.spatialscores.fss_accum(
                        fss=self.tables[name], X_f=x_pred[i], X_o=x_obs[i]
                    )
        self.is_empty = False

    def compute(self):
        names = []
        values = []
        for scl in self.scales:
            for thr in self.thresholds:
                thr_str = str(thr).replace(".", "_")
                out_name = "FSS_{scale}_{thr}".format(scale=scl, thr=thr_str)
                names.append(out_name)
                metric_values = np.empty(len(self.leadtimes))
                for i, lt in enumerate(self.leadtimes):
                    in_name = self.name_template.format(
                        scale=scl, thresh=thr, leadtime=lt
                    )
                    metric_values[i] = verification.fss_compute(self.tables[in_name])
                values.append(metric_values)
        return np.array(values), names

    def merge(self, fss_other):
        self.tables = {
            name: verification.spatialscores.fss_merge(table, fss_other.tables[name])
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
        fss_kwargs : dict = {}
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
            **fss_kwargs
        )
