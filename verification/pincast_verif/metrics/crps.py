
import numpy as np
from pysteps import verification

from pincast_verif.metrics import Metric
from pincast_verif.plot_tools import plot_1d

class Crps(Metric):

    def __init__(
        self,
        leadtimes,
        tables : dict = None,
        **kwargs) -> None:
        
        self.name_template = "CRPS_l_{leadtime}"
        self.leadtimes = leadtimes
        if tables is None:
            self.tables = {}
            for lt in leadtimes:
                name = self.name_template.format(leadtime=lt)
                self.tables[name] = verification.probscores.CRPS_init()
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
            verification.probscores.CRPS_accum(
                CRPS = self.tables[name],
                X_f = x_pred[i],
                X_o = x_obs[i]
            )
        self.is_empty = False 

    def compute(self):
        names = [] 
        values = [] 
        out_name = "CRPS"
        names.append(out_name)
        metric_values = np.empty(len(self.leadtimes))
        for i,lt in enumerate(self.leadtimes):
            in_name = self.name_template.format(leadtime=lt)
            metric_values[i] = verification.probscores.CRPS_compute(self.tables[in_name])
        values.append(metric_values)
        return np.array(values), names

    @staticmethod
    def merge_tables(table_self, table_other):
        "CRPS tables do not have a merge function, but contain only 'n' and 'sum' fields."
        return {
            key : table_self[key] + table_other[key] for key in table_self.keys()
        }

    def merge(self, crps_other):
        self.tables = {name : Crps.merge_tables(table, crps_other.tables[name])
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
        crps_kwargs: dict = {},
        ):
        return plot_1d(
            scores,
            method,
            lt,
            exp_id,
            path_save,
            method_plot_params = method_plot_params,
            subplot_kwargs= subplot_kwargs, 
            plot_kwargs  = plot_kwargs,
            **crps_kwargs
            )
