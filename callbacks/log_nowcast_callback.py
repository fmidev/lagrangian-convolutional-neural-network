"""
Log nowcast images to the model logger.
Bent Harnist (FMI) 2022
"""
from pytorch_lightning.callbacks import Callback
from utils import radar_image_plots

class LogNowcast(Callback):
    """
    Log nowcast images to the model logger.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.verif_display = config.verif_display
        self.train_display = config.train_display


    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs : dict,
        batch : tuple,
        batch_idx: int,
        ) -> None:

        y_hat = outputs["prediction"]
        #total_loss = outputs["loss"]
        x,y,idx = batch
        to_dBZ = trainer.datamodule.train_dataset.from_transformed
        #pl_module.logger.experiment.add_scalars(
        #    "losses", {"train_loss": total_loss}, global_step=pl_module.global_step)
        #pl_module.logger.experiment.add_scalars(
        #    "batch_index", {"index": batch_idx[0]}, global_step=pl_module.global_step)
        if (pl_module.global_step % self.train_display) == 0:
            time_window = trainer.datamodule.train_dataset.get_window(idx[0])
            fig = radar_image_plots.debug_training_plot(
                pl_module.global_step,
                batch_idx,
                time_window,
                to_dBZ(x),
                to_dBZ(y_hat),
                to_dBZ(y),
                write_fig=False,
            )
            pl_module.logger.experiment.add_figure(
                "train_images", fig, global_step=pl_module.global_step)
        pl_module.logger.experiment.flush()
        

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: dict,
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        y_hat = outputs["prediction"]
        x,y,idx = batch
        to_dBZ = trainer.datamodule.valid_dataset.from_transformed
        if batch_idx in self.verif_display:
            time_window = trainer.datamodule.valid_dataset.get_window(idx[0])
            fig = radar_image_plots.debug_training_plot(
                pl_module.global_step,
                batch_idx,
                time_window,
                to_dBZ(x),
                to_dBZ(y_hat),
                to_dBZ(y),
                write_fig=False,
            )
            pl_module.logger.experiment.add_figure(
                "valid_images", fig, global_step=pl_module.global_step)
        pl_module.logger.experiment.flush()


    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: dict,
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
    
        y_hat = outputs["prediction"]
        x,y,idx = batch
        to_dBZ = trainer.datamodule.test_dataset.from_transformed
        if batch_idx in self.verif_display:
            time_window = trainer.datamodule.test_dataset.get_window(idx[0])
            fig = radar_image_plots.debug_training_plot(
                self.global_step,
                batch_idx,
                time_window,
                to_dBZ(x),
                to_dBZ(y_hat),
                to_dBZ(y),
                write_fig=False,
            )
            pl_module.logger.experiment.add_figure(
                "test_images", fig, global_step=pl_module.global_step)
        pl_module.logger.experiment.flush()