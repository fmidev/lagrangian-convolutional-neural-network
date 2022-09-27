"""SSIM verification metric."""
from skimage.metrics import structural_similarity as ssim
import numpy as np

from pincast_verif.metrics import Metric
from pincast_verif.plot_tools import plot_1d


class SSIMMetric(Metric):
    """Metric class for SSIM."""

    def __init__(
        self,
        leadtimes: list,
        data_range: float = 1.0,
        win_size: int = 11,
        gaussian_weights: bool = False,
        tables: dict = None,
    ) -> None:
        """Initialize metric."""
        self.leadtimes = leadtimes
        self.data_range = data_range
        self.win_size = win_size
        self.gaussian_weights = gaussian_weights

        self.name_template = "SSIM"
        self.name = self.name_template

        if tables is None:
            self.tables = {self.name: {}}
            self.tables[self.name].update({"n": 0})
            self.tables[self.name].update(
                {
                    "values": np.zeros((len(self.leadtimes),)),
                }
            )
            self.is_empty = True
        else:
            self.tables = tables
            self.is_empty = False

    def accumulate(self, x_pred: np.ndarray, x_obs: np.ndarray) -> dict:
        """Accumulation function."""
        if not isinstance(x_pred, np.ndarray):
            x_pred = np.array(x_pred)
        if x_pred.ndim == 4:
            x_pred = x_pred.mean(axis=1)
        x_pred[~np.isfinite(x_pred)] = 0.0
        x_obs[~np.isfinite(x_obs)] = 0.0

        for i, lt in enumerate(self.leadtimes):
            result = ssim(
                x_pred[lt - 1],
                x_obs[lt - 1],
                data_range=self.data_range,
                win_size=self.win_size,
                gaussian_weights=self.gaussian_weights,
            )

            self.tables[self.name]["values"][i] += result

        self.tables[self.name]["n"] += 1
        self.is_empty = False

    def compute(self) -> np.ndarray:
        """Compute result."""
        names = []

        # Forecast
        names.append(self.name)
        values = self.tables[self.name]["values"] / self.tables[self.name]["n"]

        return values[np.newaxis, ...], names

    def _merge_tables(self, table_self, table_other):
        """Merge tables."""
        return {key: (table_self[key] + table_other[key]) for key in table_self.keys()}

    def merge(self, other_ssim):
        """Merge instances."""
        self.tables = {
            self.name: self._merge_tables(
                self.tables[self.name], other_ssim.tables[other_ssim.name]
            )
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
        ssim_kwargs: dict = {},
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
            **ssim_kwargs
        )
