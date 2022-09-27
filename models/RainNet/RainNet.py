"""RainNet model definition."""

from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from metrics import MAE, ETS

from networks import RainNet as RN
from utils import radar_image_plots, verification_score_plots
from costfunctions import *


class RainNet(pl.LightningModule):
    """Model for the RainNet neural network."""

    def __init__(self, config):

        super().__init__()

        self.input_shape = config.model.rainnet.input_shape

        self.network = RN(
            kernel_size=config.model.rainnet.kernel_size,
            mode=config.model.rainnet.mode,
            im_shape=self.input_shape[1:],  # x,y
            conv_shape=config.model.rainnet.conv_shape,
        )

        if config.model.loss.name == "log_cosh":
            self.criterion = LogCoshLoss()
        elif config.model.loss.name == "ssim":
            self.criterion = SSIM(**config.model.loss.kwargs)
        elif config.model.loss.name == "ms_ssim":
            self.criterion = MS_SSIM(**config.model.loss.kwargs)
        else:
            raise NotImplementedError(f"Loss {config.model.loss.name} not implemented!")

        self.n_leadtimes = config.model.n_leadtimes

        discount_rate = config.model.loss.discount_rate
        t = np.arange(1, self.n_leadtimes + 1)
        self.loss_weights = t ** (-discount_rate) / (t ** (-discount_rate)).sum()

        if self.n_leadtimes > 1:
            self.automatic_optimization = False

        self.thresholds = config.model.intensity_thresholds
        self.display_step = config.model.display
        self.save_metrics = config.train_params.save_metrics
        self.lr = config.model.lr

        self.valid_scores = MetricCollection({})
        self.valid_scores.add_metrics(
            {
                f"ETS_{thr}": ETS(
                    threshold=thr, length=self.n_leadtimes, reduce_dims=(0, 2, 3)
                )
                for thr in self.thresholds
            }
        )
        self.valid_scores.add_metrics(
            {"MAE": MAE(length=self.n_leadtimes, reduce_dims=(0, 2, 3))}
        )

        self.train_scores = self.valid_scores.clone()
        self.test_scores = self.valid_scores.clone()

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        # learning rate from Ayzel(2020)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):

        x,y,idx = batch
        if self.n_leadtimes == 1:
            y_seq = self(x)
            total_loss = self.criterion(y_seq, y[:, 0, :, :].clone())
        else:
            y_seq = torch.empty((x.shape[0], self.n_leadtimes, *self.input_shape[1:]))
            opt = self.optimizers()
            for i in range(self.n_leadtimes):
                y_hat = self(x)
                y_i = y[:, i, :, :].clone()
                loss = self.criterion(y_hat, y_i) * self.loss_weights[i]

                self.manual_backward(loss, retain_graph=True)

                del y_i
                y_seq[:, i, :, :] = y_hat.detach().squeeze()
                if i != self.n_leadtimes - 1:
                    x = torch.roll(x, -1, dims=1)
                    x[:, 3, :, :] = y_hat.squeeze()
                    del y_hat
            opt.step()
            opt.zero_grad()
            del (loss, y_hat)
            with torch.no_grad():
                total_loss = self.criterion(y[:, : self.n_leadtimes, :, :], y_seq)

        self.log("train loss", total_loss)
        to_dBZ = self.trainer.datamodule.train_dataset.from_transformed
        self.logger.experiment.add_scalars(
            "batch_index", {"index": idx[0]}, global_step=self.global_step
        )
        if (self.global_step % self.display_step) == 0:
            time_window = self.trainer.datamodule.train_dataset.get_window(idx[0])
            fig = radar_image_plots.debug_training_plot(
                self.global_step,
                batch_idx,
                time_window,
                to_dBZ(x),
                to_dBZ(y_seq),
                to_dBZ(y),
                write_fig=False,
            )
            self.logger.experiment.add_figure(
                "output_images", fig, global_step=self.global_step
            )
        self.logger.experiment.add_scalars(
            "losses", {"train_loss": total_loss}, global_step=self.global_step
        )
        self.logger.experiment.flush()

        if self.automatic_optimization:
            return total_loss
        else:
            return

    def training_epoch_end(self, outputs):
        return

    def validation_step(self, batch, batch_idx):

        x, y, idx = batch
        if self.n_leadtimes == 1:
            y_hat = self(x)
            total_loss = self.criterion(y_hat, y[:, 0, :, :].clone())
        else:
            y_hat = torch.empty((x.shape[0], self.n_leadtimes, *self.input_shape[1:]))
            for i in range(self.n_leadtimes):
                pred = self(x)
                y_hat[:, i, :, :] = pred.detach().squeeze()
                if i != self.n_leadtimes - 1:
                    x = torch.roll(x, -1, dims=1)
                    x[:, 3, :, :] = pred.squeeze()
                    del pred
            total_loss = self.criterion(y[:, : self.n_leadtimes, :, :], y_hat)

        self.log("val_loss", total_loss)

        to_dBZ = self.trainer.datamodule.valid_dataset.from_transformed
        # self.valid_scores.update(to_dBZ(y_hat_sequence), to_dBZ(y))

        if (self.global_step % self.display_step) == 0:
            time_window = self.trainer.datamodule.valid_dataset.get_window(idx[0])
            fig = radar_image_plots.debug_training_plot(
                self.global_step,
                batch_idx,
                time_window,
                to_dBZ(x),
                to_dBZ(y_hat),
                to_dBZ(y),
                write_fig=False,
            )
            self.logger.experiment.add_figure(
                "valid_images", fig, global_step=self.global_step
            )
        self.logger.experiment.flush()

    def validation_epoch_end(self, outputs):
        """Calculate custom metrics for the validation data."""
        return self._epoch_end("valid", outputs)

    def test_step(self, batch, batch_idx):
        x, y, idx = batch
        if self.n_leadtimes == 1:
            y_hat = self(x)
            total_loss = self.criterion(y_hat, y[:, 0, :, :].clone())
        else:
            y_hat = torch.empty((1, self.n_leadtimes, *self.input_shape[1:]))
            for i in range(self.n_leadtimes):
                pred = self(x)
                y_hat[:, i, :, :] = pred.detach()
                if i != self.n_leadtimes - 1:
                    x = torch.roll(x, -1, dims=1)
                    x[:, 3, :, :] = pred.squeeze()
                    del pred
            total_loss = self.criterion(y[:, : self.n_leadtimes, :, :], y_hat)

        self.log("val_loss", total_loss)
        assert y_hat.shape == y.shape
        to_dBZ = self.trainer.datamodule.valid_dataset.from_transformed
        self.valid_scores.update(to_dBZ(y_hat), to_dBZ(y))

        if (self.global_step % self.display_step) == 0:
            time_window = self.trainer.datamodule.test_dataset.get_window(idx[0])
            fig = radar_image_plots.debug_training_plot(
                self.global_step,
                batch_idx,
                time_window,
                to_dBZ(x),
                to_dBZ(y_hat),
                to_dBZ(y),
                write_fig=False,
            )
            self.logger.experiment.add_figure(
                "test_images", fig, global_step=self.global_step
            )
        self.logger.experiment.flush()

    def test_epoch_end(self, outputs):
        """Calculate custom metrics for the validation data."""
        return self._epoch_end("test", outputs)

    def _epoch_end(self, stage, outputs):
        """Calculate custom metrics for the training data."""

        if not self.save_metrics:
            self.logger.experiment.flush()
            return

        if stage == "train":
            scores = self.train_scores.compute()
        elif stage == "valid":
            scores = self.valid_scores.compute()
        elif stage == "test":
            scores = self.test_scores.compute()

        cat_scores = defaultdict(dict)
        cont_scores = defaultdict(dict)

        for il in range(self.n_leadtimes):
            lt = 5 * (il + 1)
            for thr in self.thresholds:
                if self.n_leadtimes > 1:
                    cat_scores[lt][thr] = {
                        "ETS": scores[f"ETS_{thr}"][il].detach().numpy()
                    }
                else:
                    cat_scores[lt][thr] = {"ETS": scores[f"ETS_{thr}"].item()}
            # Continuous scores
            if self.n_leadtimes > 1:
                cont_scores[lt] = {"MAE": scores[f"MAE"][il].detach().numpy()}
            else:
                cont_scores[lt] = {"MAE": scores[f"MAE"].item()}

        cat_df = pd.concat({k: pd.DataFrame(v).T for k, v in cat_scores.items()})
        cat_df["Leadtime"] = cat_df.index.get_level_values(0)
        cat_df["Threshold"] = cat_df.index.get_level_values(1)

        # categorical scores
        """
        fig = verification_score_plots.plot_cat_scores_against_leadtime(
            cat_df, f"{self.current_epoch:03d}_{stage}_cat_scores.png")
        self.logger.experiment.add_figure(
            f"{stage}_cat_scores", fig, global_step=self.current_epoch)
        """
        cat_df.to_csv(
            f"{stage}_cat_scores_{self.current_epoch}_{self.logger.version}.csv"
        )

        # Continuous scores
        cont_df = pd.DataFrame(cont_scores).T
        cont_df["Leadtime"] = cont_df.index.values
        cont_df.to_csv(
            f"{stage}_cont_scores_" + f"{self.current_epoch}_{self.logger.version}.csv"
        )

        """
        fig = verification_score_plots.plot_cont_scores_against_leadtime(
            cont_df, f"{self.current_epoch:03d}_{stage}_cont_scores.png")
        self.logger.experiment.add_figure(
            f"{stage}_cont_scores", fig, global_step=self.current_epoch)
        """
        # self.logger_name format deleted
        self.logger.experiment.flush()


