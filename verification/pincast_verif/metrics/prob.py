import numpy as np
from pysteps import verification
import matplotlib.pyplot as plt
from pincast_verif.metrics import Metric
from pysteps.postprocessing.ensemblestats import excprob


class ProbabilisticMetric(Metric):
    "Probabilistic forecast metrics (for ensemble based forecasts): ROC and RELDIAG"
    def __init__(
        self,
        leadtimes : list,
        prob_metrics : dict,
        thresholds : list,
        tables : dict = None,
        ):
        super().__init__()
        self.name_template = "{metric}_l_{leadtime}_t_{thresh}"
        self.leadtimes = leadtimes
        self.prob_metrics = prob_metrics
        self.thresholds = thresholds
        if tables is None:
            self.tables = {}
            for metric, metric_args in self.prob_metrics.items():
                for lt in metric_args.leadtimes:
                    for thresh in self.thresholds:
                        name = self.name_template.format(leadtime=lt, metric=metric, thresh=thresh)
                        if metric == "ROC":
                            self.tables[name] = verification.ROC_curve_init(X_min=thresh, **metric_args.kwargs)
                        elif metric == "RELDIAG":
                            self.tables[name] == verification.reldiag_init(X_min=thresh, **metric_args.kwargs)
                        else:
                            raise ValueError(f"Invalid metric name : {metric}")
            self.is_empty = True
        else:
            self.tables = tables
            self.is_empty = False

    def accumulate(self, x_pred, x_obs):
        if x_pred.ndim != 4:
            raise ValueError(f"Prediction array must be 4-dimensional with (T,S,W,H), but instead is of shape {x_pred.shape}")
        probs = np.stack(excprob(X=x, X_thr=self.thresholds) for x in x_pred)
        for metric, metric_args in self.prob_metrics.items():
                for i,lt in enumerate(metric_args.leadtimes):
                    for j, thresh in enumerate(self.thresholds):
                        name = self.name_template.format(leadtime=lt, metric=metric, thresh=thresh)
                        if metric == "ROC":
                            verification.ROC_curve_accum(self.tables[name], P_f=probs[i,j], X_o=x_obs[i])
                        elif metric == "RELDIAG":
                            verification.reldiag_accum(self.tables[name], P_f=probs[i,j], X_o=x_obs[i])
                        else:
                            raise ValueError(f"Invalid metric name : {metric}")
        self.is_empty = False 

    
    def compute(self):
        names = [] 
        values = [] 
        for metric, metric_args in self.prob_metrics.items():
                for i,lt in enumerate(metric_args.leadtimes):
                    for j, thresh in enumerate(self.thresholds):
                        name = self.name_template.format(leadtime=lt, metric=metric, thresh=thresh)
                        if metric == "ROC":
                            POD, POFD, area = verification.ROC_curve_compute(self.tables[name], compute_area=True)
                            data_to_add = np.stack([POD, POFD, area*np.ones(len(POD))])
                        elif metric == "RELDIAG":
                            x,y = verification.reldiag_compute(self.tables[name])
                            data_to_add = np.stack([x, y, np.asarray([None]*len(x))])
                        else:
                            raise ValueError(f"Invalid metric name : {metric}")
                        names.append(name)
                        values.append(data_to_add)

        return np.array(values), names

    @staticmethod
    def merge_tables(table_self, table_other):
        "ROC, Reliability diag. tables do not have a built-in merge function either.. :("
        return {
            key : (table_self[key] + table_other[key] 
            if key in ["hits", "misses", "false_alarms",
            "corr_neg", "X_sum", "Y_sum", "num_idx", "sample_size"] 
            else table_self[key]) # prob_thrs, X_min, bin_edges, n_bins, min_count
            for key in table_self.keys()
        }

    def merge(self, prob_other):
        self.tables = {name : ProbabilisticMetric.merge_tables(table, prob_other.tables[name])
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
        prob_kwargs: dict = {},
        ):
        if method == "ALL":
            raise NotImplementedError(f"Method ALL plotting not implemented for probabilistic forecasts.")
        pass
        """
        So here...

        Plot ROC area under the curve for method ALL or other (rainnet, steps,..) 
        
        """