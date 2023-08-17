"""Log nowcast images to the model logger.

Bent Harnist (FMI) 2022
Jenna Ritvanen (FMI) 2023

"""
import numpy as np
from pytorch_lightning.callbacks import Callback

from utils import plot_input_pred_multifeature_array


class LogDiffNowcast(Callback):
    """Log nowcast images to the model logger."""

    def __init__(self, config) -> None:
        """Initialize the callback."""
        super().__init__()
        self.verif_display = config.verif_display
        self.train_display = config.train_display
        self.plot_batch_indices = config.plot_batch_indices
        self.plot_conf = config.plot_conf

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: dict,
        batch: tuple,
        batch_idx: int,
    ) -> None:
        """Log training images to the model logger at batch end.

        Parameters
        ----------
        trainer : pl.Trainer
            Trainer instance
        pl_module : pl.LightningModule
            Lightning module
        outputs : dict
            Module outputs
        batch : tuple
            Input batch
        batch_idx : int
            Batch index

        """
        if (pl_module.global_step % self.train_display) == 0:
            x, y, idx = batch
            y_hat = outputs["prediction"]
            # x = batch["inputs"]
            # y = batch["targets"]
            # idx = batch["idx"]
            x_ = trainer.datamodule.train_dataset.apply_denormalization(
                x, feature_axis=-4
            )
            y_hat_ = trainer.datamodule.train_dataset.apply_denormalization(
                y_hat, feature_axis=-4, target=True
            )
            conf = {
                i: self.plot_conf[n]
                for i, n in enumerate(trainer.datamodule.train_dataset.feature_names)
            }
            datewindow = trainer.datamodule.train_dataset.get_window(
                int(idx[0]), in_datetime=True
            )

            fig = plot_input_pred_multifeature_array(
                x_[0, ...].detach().cpu().numpy()[np.newaxis, ...],
                y_hat_[0, ...].detach().cpu().numpy()[np.newaxis, ...],
                conf,
                input_times=datewindow[
                    : trainer.datamodule.train_dataset.input_block_length
                ],
                pred_times=datewindow[
                    trainer.datamodule.train_dataset.input_block_length :
                ],
            )
            trainer.logger.experiment.add_figure(
                "train_images", fig, global_step=pl_module.global_step
            )
            trainer.logger.experiment.flush()