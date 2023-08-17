"""Lagrangian CNN (L-CNN) model definition."""

import numpy as np
import torch
import pytorch_lightning as pl
from scipy.ndimage import uniform_filter
from functools import partial

from networks import RainNet as RN
from utils import (
    read_advection_fields_from_h5,
    transform_to_eulerian,
    read_advection_fields_from_nc,
    rainrate_to_dbz,
)
from costfunctions import *


class LCNN(pl.LightningModule):
    """Model for the Lagrangian CNN (L-CNN) neural network."""

    def __init__(self, config):

        super().__init__()
        self.save_hyperparameters()

        self.input_shape = config.model.rainnet.input_shape
        self.personal_device = torch.device(config.train_params.device)
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
        elif config.model.loss.name == "rmse":
            self.criterion = RMSELoss()
        else:
            raise NotImplementedError(f"Loss {config.model.loss.name} not implemented!")

        # on which leadtime to train the NN on?
        self.train_leadtimes = config.model.train_leadtimes
        self.verif_leadtimes = config.train_params.verif_leadtimes
        # How many leadtimes to predict
        self.predict_leadtimes = config.prediction.predict_leadtimes
        self.predict_extrap_kwargs = config.prediction.extrap_kwargs
        self.euler_transform_nworkers = config.prediction.euler_transform_nworkers
        if "zr_a" in config.prediction_output and "zr_b" in config.prediction_output:
            self.zr_a = config.prediction_output.zr_a
            self.zr_b = config.prediction_output.zr_b
        else:
            self.zr_a = 223.0
            self.zr_b = 1.53

        # 1.0 corresponds to harmonic loss weight decrease,
        # 0.0 to no decrease at all,
        # less than 1.0 is sub-harmonic,
        # more is super-harmonic
        discount_rate = config.model.loss.discount_rate
        # equal weighting for each lt, sum to one.
        if discount_rate == 0:
            self.train_loss_weights = (
                np.ones(self.train_leadtimes) / self.train_leadtimes
            )
            self.verif_loss_weights = (
                np.ones(self.verif_leadtimes) / self.verif_leadtimes
            )
        # Diminishing weight by n_lt^( - discount_rate), sum to one.
        else:
            train_t = np.arange(1, self.train_leadtimes + 1)
            self.train_loss_weights = (
                train_t ** (-discount_rate) / (train_t ** (-discount_rate)).sum()
            )
            verif_t = np.arange(1, self.verif_leadtimes + 1)
            self.verif_loss_weights = (
                verif_t ** (-discount_rate) / (verif_t ** (-discount_rate)).sum()
            )

        # optimization parameters
        self.lr = float(config.model.lr)
        self.lr_sch_params = config.train_params.lr_scheduler
        self.automatic_optimization = False

        # Whether to apply differencing
        self.apply_differencing = config.model.apply_differencing

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_sch_params.name is None:
            return optimizer
        elif self.lr_sch_params.name == "reduce_lr_on_plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, **self.lr_sch_params.kwargs
            )
            return [optimizer], [lr_scheduler]
        else:
            raise NotImplementedError("Lr scheduler not defined.")

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        y_hat, total_loss = self._iterative_prediction(batch=batch, stage="train")
        opt.step()
        opt.zero_grad()
        self.log("train_loss", total_loss)
        return {"prediction": y_hat, "loss": total_loss}

    def validation_step(self, batch, batch_idx):
        y_hat, total_loss = self._iterative_prediction(batch=batch, stage="valid")
        self.log("val_loss", total_loss, sync_dist=True)
        return {"prediction": y_hat, "loss": total_loss}

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])

    def test_step(self, batch, batch_idx):
        y_hat, total_loss = self._iterative_prediction(batch=batch, stage="test")
        self.log("test_loss", total_loss)
        return {"prediction": y_hat, "loss": total_loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Get data
        if self.apply_differencing:
            x, y, idx, first_x = batch
        else:
            x, y, idx = batch

        # Perform prediction with LCNN model
        x_ = x.clone()
        y_seq = self._iterative_prediction(batch=(x, y, idx), stage="predict")

        # Transform from scaled to mm/hh
        if self.trainer.datamodule.predict_dataset.__class__.__name__ == "LagrangianFMICompositeDataset":
            invScaler = self.trainer.datamodule.predict_dataset.invScaler
        elif self.trainer.datamodule.predict_dataset.__class__.__name__ == "LagrangianSwissCompositeDataset":
            invScaler = self.trainer.datamodule.predict_dataset.apply_denormalization
        else:
            raise NotImplementedError(f"Dataset {self.trainer.datamodule.predit_dataset} not implemented")
        y_seq = invScaler(y_seq, target=True, feature_axis=-4)
        x = invScaler(x_, target=False, feature_axis=-4)

        # If we applied differencing, integrate back to full fields
        if self.apply_differencing:
            # This should be a dummy transform that has no effect (since scaling should be "none"),
            # but keep to match what is done to predictions
            first_x = invScaler(first_x, target=False, feature_axis=-3)
            # Integrate observation back to full fields
            x[:, 0, :, :] = first_x + x[:, 0, :, :]
            for i in range(1, x.shape[1]):
                x[:, i, :, :] = x[:, i, :, :] + x[:, i - 1, :, :]
            # First prediction with last observation
            y_seq[:, 0, :, :] = y_seq[:, 0, :, :] + x[:, -1, :, :]
            # Iterate through rest of predictions
            for i in range(1, self.predict_leadtimes):
                y_seq[:, i, :, :] = y_seq[:, i - 1, :, :] + y_seq[:, i, :, :]

            # Set negative values (in mm/h) to 0
            y_seq[y_seq < 0] = 0

        # Extrapolate the predictions to correct time
        output_fields = torch.empty_like(y_seq)
        for batch_idx in range(x.shape[0]):
            # Get commontime
            common_time = self.trainer.datamodule.predict_dataset.get_common_time(
                idx[batch_idx], in_datetime=True
            )
            window = self.trainer.datamodule.predict_dataset.get_window(idx[batch_idx], in_datetime=True)

            # If differencing, the first time is dropped
            if self.apply_differencing:
                # Account for the first input image which is removed
                window = window[1:]
                common_time_index = (
                    self.trainer.datamodule.predict_dataset.common_time_index
                )
            else:
                common_time_index = (
                    self.trainer.datamodule.predict_dataset.common_time_index + 1
                )

            if self.trainer.datamodule.predict_dataset.__class__.__name__ == "LagrangianFMICompositeDataset":
                filename = self.trainer.datamodule.predict_dataset.get_filename(common_time)
                advfields = read_advection_fields_from_h5(filename)
            elif self.trainer.datamodule.predict_dataset.__class__.__name__ == "LagrangianSwissCompositeDataset":
                filename = common_time.strftime(self.trainer.datamodule.predict_dataset.advection_file_path)
                advfields = read_advection_fields_from_nc(filename)

            # Apply bbox and downsample advfields
            downsample = self.trainer.datamodule.predict_dataset.downsample_data
            bbox = self.trainer.datamodule.predict_dataset.apply_bbox
            for k, field in advfields.items():
                bbox_field = bbox(field)

                if bbox_field.shape[1] != x.shape[2]:
                    x_field = downsample(bbox_field[0, ...])
                    y_field = downsample(bbox_field[1, ...])

                    factor_x = int(bbox_field.shape[1] / x_field.shape[0])
                    factor_y = int(bbox_field.shape[2] / x_field.shape[1])
                    advfields[k] = np.stack([x_field / factor_x, y_field / factor_y])
                else:
                    advfields[k] = bbox_field

            # Transform to Eulerian fields
            lagrangian_fields = np.vstack(
                [
                    x[batch_idx, ...].detach().cpu(),
                    y_seq[batch_idx, ...].detach().cpu().squeeze(),
                ]
            )
            euler_fields = transform_to_eulerian(
                lagrangian_fields,
                common_time_index,
                window,
                advfields,
                extrap_kwargs=self.predict_extrap_kwargs,
                prediction_mode=True,
                n_workers=self.euler_transform_nworkers,
            )

            # Fill possible negative values caused by interpolation with moving mean
            if np.any(euler_fields < 0):
                nan_mask = np.isnan(euler_fields)
                euler_fields[nan_mask] = 0
                for i in range(euler_fields.shape[0]):
                    mean_ = uniform_filter(euler_fields[i], size=3, mode="constant")
                    euler_fields[i][np.where(euler_fields[i] < 0)] = mean_[
                        np.where(euler_fields[i] < 0)
                    ]

                # If any negative remain, set to 0
                euler_fields[euler_fields < 0] = 0
                euler_fields[nan_mask] = np.nan

            # output_fields[batch_idx, ...] = torch.from_numpy(
            #     euler_fields[x.shape[1] :, ...]
            # )
        # Transform from mm/h to dBZ
        euler_fields = rainrate_to_dbz(euler_fields, zr_a=self.zr_a, zr_b=self.zr_b)
        output_fields[batch_idx, ...] = torch.from_numpy(
            euler_fields[x_.shape[1] :, ...]
        )
        del x, y_seq, lagrangian_fields, euler_fields, advfields
        return output_fields

    def _iterative_prediction(self, batch, stage):

        if stage == "train":
            n_leadtimes = self.train_leadtimes
            calculate_loss = True
            loss_weights = self.train_loss_weights
        elif stage == "valid" or stage == "test":
            n_leadtimes = self.verif_leadtimes
            calculate_loss = True
            loss_weights = self.verif_loss_weights
        elif stage == "predict":
            n_leadtimes = self.predict_leadtimes
            calculate_loss = False
        else:
            raise ValueError(
                f"Stage {stage} is undefined. \n choices: 'train', 'valid', test', 'predict'"
            )

        x, y, _ = batch
        y_seq = torch.empty(
            (x.shape[0], n_leadtimes, *self.input_shape[1:]), device=self.device
        )
        if calculate_loss:
            total_loss = 0

        for i in range(n_leadtimes):
            y_hat = self(x)
            if calculate_loss:
                y_i = y[:, None, i, :, :].clone()
                loss = self.criterion(y_hat, y_i) * loss_weights[i]
                total_loss += loss.detach()
                if stage == "train":
                    self.manual_backward(loss)
                del y_i
            y_seq[:, i, :, :] = y_hat.detach().squeeze()
            if i != n_leadtimes - 1:
                x = torch.roll(x, -1, dims=1)
                x[:, 3, :, :] = y_hat.detach().squeeze()
            del y_hat
        if calculate_loss:
            return y_seq, total_loss
        else:
            return y_seq
