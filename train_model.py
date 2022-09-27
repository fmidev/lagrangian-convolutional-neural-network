# import os
import sys
from pathlib import Path
import argparse
import random
import numpy as np

from utils.config import load_config
from utils.logging import setup_logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    DeviceStatsMonitor,
)
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profiler import PyTorchProfiler

from datamodules import LagrangianFMICompositeDataModule

from datasets import LagrangianFMIComposite

from models import LCNN

from callbacks import NowcastMetrics, LogNowcast


def main(configpath, checkpoint=None):
    confpath = Path("config") / configpath
    dsconf = load_config(confpath / "lagrangian_datasets.yaml")
    outputconf = load_config(confpath / "output.yaml")
    modelconf = load_config(confpath / "lcnn.yaml")
    metricsconf = load_config(confpath / "nowcast_metrics_callback.yaml")
    lognowcastconf = load_config(confpath / "log_nowcast_callback.yaml")

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    setup_logging(outputconf.logging)

    datamodel = LagrangianFMICompositeDataModule(dsconf, modelconf.train_params)

    model = LCNN(modelconf)

    # Callbacks
    model_ckpt = ModelCheckpoint(
        dirpath=f"checkpoints/{modelconf.train_params.savefile}",
        save_top_k=3,
        monitor="val_loss",
        save_on_train_epoch_end=False,
    )
    nowcast_image_logger = LogNowcast(config=lognowcastconf)
    nowcast_metrics = NowcastMetrics(config=metricsconf)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stopping = EarlyStopping(**modelconf.train_params.early_stopping)
    # device_monitor = DeviceStatsMonitor()
    tb_logger = pl_loggers.TensorBoardLogger("logs", name=f"train_{configpath}")
    profiler = PyTorchProfiler(profile_memory=False)

    trainer = pl.Trainer(
        profiler=profiler,
        logger=tb_logger,
        val_check_interval=modelconf.train_params.val_check_interval,
        max_epochs=modelconf.train_params.max_epochs,
        max_time=modelconf.train_params.max_time,
        gpus=modelconf.train_params.gpus,
        limit_val_batches=modelconf.train_params.val_batches,
        limit_train_batches=modelconf.train_params.train_batches,
        callbacks=[
            early_stopping,
            model_ckpt,
            lr_monitor,
            nowcast_image_logger,
            nowcast_metrics,
        ],
    )

    trainer.fit(model=model, datamodule=datamodel, ckpt_path=checkpoint)

    torch.save(model.state_dict(), f"state_dict_{modelconf.train_params.savefile}.ckpt")
    trainer.save_checkpoint(f"{modelconf.train_params.savefile}.ckpt")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("config", type=str, help="Configuration folder")
    argparser.add_argument(
        "-c",
        "--continue_training",
        type=str,
        default=None,
        help="Path to checkpoint for model that is continued.",
    )
    args = argparser.parse_args()
    main(args.config, args.continue_training)
