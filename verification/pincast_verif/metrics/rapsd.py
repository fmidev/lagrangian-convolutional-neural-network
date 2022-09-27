import numpy as np
from pysteps.utils.spectral import rapsd
from pysteps.visualization.spectral import plot_spectrum1d
from pysteps.utils.transformation import dB_transform
import matplotlib.pyplot as plt
import seaborn as sns

from pincast_verif.metrics import Metric
from pincast_verif.plot_tools import plot_rapsd


class RapsdMetric(Metric):
    def __init__(
        self,
        leadtimes: list,
        im_size: tuple = (512, 512),
        fft_method: callable = np.fft,
        return_freq: bool = True,
        normalize: bool = True,
        d: float = 1.0,
        tables: dict = None,
    ) -> None:

        self.leadtimes = leadtimes
        self.name_template = "RAPSD_l_{lts}"
        self.name = self.name_template.format(
            lts="_".join([str(lt * 5) for lt in self.leadtimes])
        )
        self.n_freqs = int(max(im_size) / 2)

        self.fft_method = fft_method
        self.normalize = normalize  # does power spectrum sum to one
        self.return_freq = return_freq
        self.d = d  # 1 / sampling rate

        if tables is None:
            self.tables = {self.name: {}}
            self.tables[self.name].update({"n": 0})
            self.tables[self.name].update(
                {
                    "values": np.zeros((len(self.leadtimes), self.n_freqs)),
                    "obs_values": np.zeros((len(self.leadtimes), self.n_freqs)),
                }
            )
            self.is_empty = True
        else:
            self.tables = tables
            self.is_empty = False

    def accumulate(self, x_pred: np.ndarray, x_obs: np.ndarray) -> dict:
        if not isinstance(x_pred, np.ndarray):
            x_pred = np.array(x_pred)
        if x_pred.ndim == 4:
            x_pred = x_pred.mean(axis=1)
        x_pred[~np.isfinite(x_pred)] = 0.0
        x_obs[~np.isfinite(x_obs)] = 0.0

        for i, lt in enumerate(self.leadtimes):
            result, freq = rapsd(
                field=x_pred[lt - 1],
                fft_method=self.fft_method,
                return_freq=True,
                normalize=self.normalize,
                d=self.d,
            )
            obs_result = rapsd(
                field=x_obs[lt - 1],
                fft_method=self.fft_method,
                return_freq=False,
                normalize=self.normalize,
                d=self.d,
            )

            self.tables[self.name]["values"][i] += result
            self.tables[self.name]["obs_values"][i] += obs_result
            if self.return_freq and "freq" not in self.tables[self.name]:
                self.tables[self.name]["freq"] = freq
        self.tables[self.name]["n"] += 1
        self.is_empty = False

    def compute(self) -> np.ndarray:
        names = []

        # Forecast
        names.append(self.name)
        values = self.tables[self.name]["values"] / self.tables[self.name]["n"]

        # Observations
        names.append("OBS_" + self.name)
        obs_values = self.tables[self.name]["obs_values"] / self.tables[self.name]["n"]
        values = np.append(
            arr=values,
            values=obs_values,
            axis=0,
        )
        if self.return_freq:
            names.append("freq")
            values = np.append(
                arr=values,
                values=self.tables[self.name]["freq"][np.newaxis, ...],
                axis=0,
            )
        return values, names

    def _merge_tables(self, table_self, table_other):
        return {
            key: (
                table_self[key] + table_other[key] if key != "freq" else table_self[key]
            )
            for key in table_self.keys()
        }

    def merge(self, other_rapsd):
        self.tables = {
            self.name: self._merge_tables(
                self.tables[self.name], other_rapsd.tables[other_rapsd.name]
            )
        }
        if (
            self.return_freq
            and "freq" not in self.tables[self.name]
            and "freq" in other_rapsd.tables[other_rapsd.name]
        ):
            self.tables[self.name]["freq"] = other_rapsd.tables[other_rapsd.name][
                "freq"
            ]

    @staticmethod
    def plot(
        data: dict,
        method: str,
        leadtimes: list,
        exp_id: str,
        path_save: str,
        kwargs: dict,
        method_plot_params: dict = {},
    ):
        plot_scales = [512, 256, 128, 64, 32, 16, 8, 4, 2]
        if method == "ALL":
            for i, lt in enumerate(leadtimes):
                fig, ax = plt.subplots()
                ax.grid(alpha=0.4, linestyle="--")
                color = sns.color_palette("husl", len(data))
                plotted_obs = False
                for (meth, (arr, name)), c in zip(data.items(), color):

                    kwarg_dict = method_plot_params[meth].copy()
                    if "color" in kwarg_dict.keys():
                        kwarg_dict.pop("color")
                    if "label" in kwarg_dict.keys():
                        kwarg_dict.pop("label")

                    if not plotted_obs:
                        plot_spectrum1d(
                            fft_freq=arr[-1],
                            fft_power=arr[len(leadtimes) + i],
                            x_units="km",
                            y_units="dBR",
                            label="Observations",
                            ax=ax,
                            color="k",
                            wavelength_ticks=plot_scales,
                            linestyle="--",
                        )
                        plotted_obs = True

                    if "label" not in method_plot_params[meth].keys():
                        method_plot_params[meth]["label"] = meth
                    if "color" not in method_plot_params[meth].keys():
                        method_plot_params[meth]["color"] = color

                    plot_spectrum1d(
                        fft_freq=arr[-1],
                        fft_power=arr[i],
                        x_units="km",
                        y_units="dBR",
                        ax=ax,
                        wavelength_ticks=plot_scales,
                        **method_plot_params[meth],
                    )
                ax.legend()
                ax.set_ylim((-50, 100))
                ax.set_title(f"RAPSD at {str(int(lt*5))} minutes")
                fig.savefig(
                    path_save.format(
                        id=exp_id,
                        method=method,
                        metric=str(int(lt * 5)) + "min_" + "RAPSD",
                    ),
                    bbox_inches="tight",
                    dpi=600,
                )
                plt.close(fig)
        else:
            fig, ax = plt.subplots()
            ax.grid(alpha=0.4, linestyle="--")
            arr, name = data[method]

            kwarg_dict = method_plot_params[method].copy()
            if "color" in kwarg_dict.keys():
                kwarg_dict.pop("color")
            if "label" in kwarg_dict.keys():
                kwarg_dict.pop("label")

            color = plt.cm.copper(np.linspace(0, 1, len(leadtimes)))
            for (i, lt), c in zip(enumerate(leadtimes), color):

                plot_spectrum1d(
                    fft_freq=arr[-1],
                    fft_power=arr[len(leadtimes) + i],
                    x_units="km",
                    y_units="dBR",
                    label=f"{int(lt*5)} minutes observations",
                    ax=ax,
                    color=c,
                    wavelength_ticks=plot_scales,
                    linestyle="--",
                )

                plot_spectrum1d(
                    fft_freq=arr[-1],
                    fft_power=arr[i],
                    x_units="km",
                    y_units="dBR",
                    label=f"{int(lt*5)} minutes nowcast",
                    ax=ax,
                    color=c,
                    wavelength_ticks=plot_scales,
                    **kwarg_dict,
                )
            ax.set_title(f"{name[0]} for method {method}")
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            fig.savefig(
                path_save.format(id=exp_id, method=method, metric=name[0]),
                bbox_inches="tight",
                dpi=600,
            )
            plt.close(fig)
