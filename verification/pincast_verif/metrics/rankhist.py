import numpy as np
from pysteps import verification
import matplotlib.pyplot as plt
from pincast_verif.metrics import Metric

class RankHistogram(Metric):
    "Rank Histogram skill metric for ensemble forecasts"
    def __init__(
        self,
        leadtimes : list,
        num_ens_member : int,
        X_min : float = None,
        tables : dict = None
    ):
        self.name_template = "rankhist_l_{leadtime}"
        self.leadtimes = leadtimes
        self.num_ens_member = num_ens_member

        if tables is None:
            self.tables = {}
            for lt in leadtimes:
                name = self.name_template.format(leadtime=lt)
                self.tables[name] = verification.rankhist_init(num_ens_members=self.num_ens_member, X_min=X_min)
            self.is_empty = True
        else:
            self.tables = tables
            self.is_empty = False

    def accumulate(self, x_pred, x_obs):
        if x_pred.ndim != 4:
            raise ValueError(f"Prediction array must be 4-dimensional with (T,S,W,H), but instead is of shape {x_pred.shape}")
        for i, lt in enumerate(self.leadtimes):
            name = self.name_template.format(
                leadtime=lt
                )
            verification.rankhist_accum(
                rankhist = self.tables[name],
                X_f = x_pred[i],
                X_o = x_obs[i]
            )
        self.is_empty = False 

    def compute(self):
        names = [] 
        values = [] 

        for lt in self.leadtimes:
            name = self.name_template.format(leadtime=lt)
            metric_values = verification.rankhist_compute(self.tables[name])
            names.append(name)
            values.append(metric_values)

        return np.array(values), names

    @staticmethod
    def merge_tables(table_self, table_other):
        "Rank histogram tables do not have a merge function"
        return {
            key : (table_self[key] + table_other[key] if key == "n" 
            else table_self[key]) # num_ens_member, X_min
            for key in table_self.keys()
        }

    def merge(self, rankhist_other):
        self.tables = {name : RankHistogram.merge_tables(table, rankhist_other.tables[name])
            for name,table in self.tables.items()}

    @staticmethod
    def plot(
        scores: dict,
        method: str,
        lt: np.ndarray,
        exp_id: str,
        path_save: str,
        method_plot_params: dict = {},
        subplot_kwargs: dict = {}, 
        plot_kwargs: dict = {},
        rankhist_kwargs: dict = {},
        ):
            if method == "ALL":
                raise NotImplementedError(f"Method ALL plotting not implemented for rank histograms.")
            for arr, names in scores[method]: #arr is (lt, num_ens_member) sized
                for i, title in enumerate(names):
                    fig, ax = plt.subplots()
                    x = np.linspace(0, 1, len(arr[i]) + 1)
                    ax.bar(x, arr[i], width=1.0 / len(x), align="edge", color="gray", edgecolor="black")

                    ax.set_xticks(x[::3] + (x[1] - x[0]))
                    ax.set_xticklabels(np.arange(1, len(x) + 1)[::3])
                    ax.set_xlim(0, 1 + 1.0 / len(x))
                    ax.set_ylim(0, np.max(arr[i]) * 1.25)

                    ax.set_xlabel("Rank of observation (among ensemble members)")
                    ax.set_ylabel("Relative frequency")

                    ax.grid(True, axis="y", ls=":")
                    path_save_now = path_save.format(id=exp_id, method=method, metric=title)
                    fig.savefig(path_save_now)
                    fig.close()


