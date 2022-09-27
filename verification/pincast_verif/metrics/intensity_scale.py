import numpy as np
import matplotlib.pyplot as plt
from pysteps import verification

from pincast_verif.metrics import Metric
from pincast_verif.plot_tools import plot_intensityscale


class IntensityScaleMetric(Metric):
    def __init__(
        self, leadtimes, thresh, scales, tables: dict = None, **kwargs
    ) -> None:

        self.name_template = "INTENSITY_SCALE_l_{leadtime}"
        self.leadtimes = leadtimes
        self.thresholds = thresh
        self.scales = scales
        if tables is None:
            self.is_empty = True
            self.tables = {}
            for lt in leadtimes:
                name = self.name_template.format(leadtime=lt)
                self.tables[name] = verification.spatialscores.intensity_scale_init(
                    name="FSS", thrs=self.thresholds, scales=self.scales
                )
        else:
            self.tables = tables
            self.is_empty = False

    def accumulate(self, x_pred, x_obs):
        if x_pred.ndim == 4:
            x_pred = x_pred.mean(axis=1)
        for i, lt in enumerate(self.leadtimes):
            name = self.name_template.format(leadtime=lt)
            verification.spatialscores.intensity_scale_accum(
                intscale=self.tables[name], X_f=x_pred[i], X_o=x_obs[i]
            )
        self.is_empty = False

    def compute(self):
        names = []
        values = np.empty((len(self.leadtimes), len(self.scales), len(self.thresholds)))
        for i, lt in enumerate(self.leadtimes):
            out_name = "INTENSITY_SCALE_l_{lt}".format(lt=lt)
            names.append(out_name)
            in_name = self.name_template.format(leadtime=lt)
            values[i] = verification.intensity_scale_compute(self.tables[in_name])
        return np.array(values), names

    def merge(self, is_other):
        self.tables = {
            name: verification.spatialscores.intensity_scale_merge(
                table, is_other.tables[name]
            )
            for name, table in self.tables.items()
        }

    @staticmethod
    def plot(
        exp_id: str,
        scores: dict,
        method: str,
        path_save_fmt: str,
        thresh: list,
        scales: list,
        kmperpixel: float,
        fig=None,
        subplot_kwargs: dict = {},
        method_plot_params: dict = {},
        vminmax=None,
    ):
        if method == "ALL":
            raise NotImplementedError(
                "IntensityScale plot not implemented for 'ALL' method"
            )
        else:
            val, names = scores[method]
            plot = {name: plt.subplots(**subplot_kwargs) for name in names}

            if not "label" in method_plot_params[method].keys():
                method_plot_params[method]["label"] = method

            for i, n in enumerate(names):
                plot_intensityscale(
                    values=val[i],
                    name=n,
                    fig=plot[n][0],
                    thresh=thresh,
                    scales=scales,
                    kmperpixel=kmperpixel,
                    unit="mm/h",
                    title=method_plot_params[method]["label"],
                    vminmax=vminmax,
                )
                path_save = path_save_fmt.format(id=exp_id, method=method, metric=n)
                plot[n][0].savefig(path_save, bbox_inches="tight", dpi=600)
                plt.close(plot[n][0])
