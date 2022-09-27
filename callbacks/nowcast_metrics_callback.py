"""
Callback to calculate nowcast skill metrics for the model attached, 
and save those metrics and their plots to the model logger / external files.
Bent Harnist (FMI) 2022
"""
from collections import defaultdict
from os import makedirs

import pandas as pd
import torch
from torchmetrics import MetricCollection
from pytorch_lightning.callbacks import Callback

from metrics import MAE, ETS
from utils import verification_score_plots

class NowcastMetrics(Callback):
    """
    Callback to calculate nowcast skill metrics for the model attached, 
    and save those metrics and their plots to the model logger / external files.
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.thresholds = config.thresholds
        self.save_metrics = config.save_metrics
        self.save_folder = config.save_folder
        if config.reduce_dims is not None:
            self.reduce_dims = config.reduce_dims
        else:
            self.reduce_dims = (0,2,3) # assuming B,T,H,W data

    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage = None
        ) -> None:

        dev = torch.device(trainer.datamodule.train_params.device) 
        self.valid_scores = MetricCollection({})
        self.valid_scores.add_metrics(
            {f"ETS_{thr}": ETS(threshold=thr, length=pl_module.verif_leadtimes, reduce_dims=self.reduce_dims).to(dev)
             for thr in self.thresholds})
        self.valid_scores.add_metrics({"MAE": MAE(length=pl_module.verif_leadtimes, reduce_dims=self.reduce_dims).to(dev)})
        self.train_scores = self.valid_scores.clone()
        self.test_scores = self.valid_scores.clone()
        if self.save_metrics:
            makedirs(name=self.save_folder, exist_ok=True)



    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: dict,
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int
        ) -> None:
        _, y, _ = batch
        y_hat = outputs["prediction"]
        to_mmh = trainer.datamodule.valid_dataset.invScaler
        self.valid_scores.update(to_mmh(y_hat), to_mmh(y))

    def on_test_batch_end(
        self, 
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: dict,
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int
        ) -> None:
        _, y, _ = batch
        y_hat = outputs["prediction"]
        to_mmh = trainer.datamodule.test_dataset.invScaler
        self.test_scores.update(to_mmh(y_hat), to_mmh(y))


    def on_validation_epoch_end(self,
    trainer: "pl.Trainer",
    pl_module: "pl.LightningModule"
    ) -> None:
        return self._epoch_end(stage="valid", trainer=trainer, pl_module=pl_module)


    def on_test_epoch_end(self,
    trainer: "pl.Trainer",
    pl_module: "pl.LightningModule"
    ) -> None:
        return self._epoch_end(stage="test", trainer=trainer, pl_module=pl_module)


    def _epoch_end(self, stage, trainer, pl_module):
        """Calculate custom metrics for the training data."""
            
        if stage == "train":
            scores = self.train_scores.compute()
        elif stage == "valid":
            scores = self.valid_scores.compute()
        elif stage == "test":
            scores = self.test_scores.compute()

        cat_scores = defaultdict(dict)
        cont_scores = defaultdict(dict)

        for il in range(pl_module.verif_leadtimes):
            lt = 5 * (il + 1)
            for thr in self.thresholds:
                if pl_module.verif_leadtimes > 1:
                    cat_scores[lt][thr] = {
                        "ETS": scores[f"ETS_{thr}"][il].detach().cpu().numpy()
                    }
                else : 
                    cat_scores[lt][thr] = {
                    "ETS": scores[f"ETS_{thr}"].item()
                }
            # Continuous scores
            if pl_module.verif_leadtimes > 1 : 
                cont_scores[lt] = {
                    "MAE": scores[f"MAE"][il].detach().cpu().numpy()
                }
            else :
                cont_scores[lt] = {
                    "MAE": scores[f"MAE"].item()
                }

        cat_df = pd.concat({k: pd.DataFrame(v).T for k, v in cat_scores.items()})
        cat_df["Leadtime"] = cat_df.index.get_level_values(0)
        cat_df["Threshold"] = cat_df.index.get_level_values(1)

        # categorical scores
        fig = verification_score_plots.plot_cat_scores_against_leadtime(
            df = cat_df,
            outfn =  f"{self.save_folder}/{pl_module.current_epoch:03d}_{stage}_cat_scores.png",
            max_leadtime=5*(pl_module.verif_leadtimes+1), write_fig=self.save_metrics)
        pl_module.logger.experiment.add_figure(
            f"{stage}_cat_scores", fig, global_step = pl_module.global_step)
        
        # Continuous scores
        cont_df = pd.DataFrame(cont_scores).T
        cont_df["Leadtime"] = cont_df.index.values
        fig = verification_score_plots.plot_cont_scores_against_leadtime(
            df = cont_df,
            outfn = f"{self.save_folder}/{pl_module.current_epoch:03d}_{stage}_cont_scores.png",
            max_leadtime=5*(pl_module.verif_leadtimes+1), write_fig=self.save_metrics
            )
        pl_module.logger.experiment.add_figure(
            f"{stage}_cont_scores", fig, global_step = pl_module.global_step)
        

        if self.save_metrics:
            cat_df.to_csv(
                f"{self.save_folder}/{stage}_cat_scores_{pl_module.current_epoch}_{pl_module.logger.version}.csv")
            cont_df.to_csv(
                f"{self.save_folder}/{stage}_cont_scores_"+
                f"{pl_module.current_epoch}_{pl_module.logger.version}.csv")

            # self.logger_name format deleted
        pl_module.logger.experiment.flush()
