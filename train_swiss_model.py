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
from pytorch_lightning.profilers import PyTorchProfiler

from datamodules import LagrangianFMICompositeDataModule

from datasets import LagrangianSwissCompositeDataModule, LagrangianSwissCompositeDataset
from models import LCNN

from callbacks import LogDiffNowcast


def main(configpath, checkpoint=None):
    confpath = Path("config") / configpath
    dsconf = load_config(confpath / "lagrangian_datasets.yaml")
    outputconf = load_config(confpath / "output.yaml")
    modelconf = load_config(confpath / "lcnn.yaml")
    lognowcastconf = load_config(confpath / "log_nowcast_callback.yaml")

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    setup_logging(outputconf.logging)

    datamodel = LagrangianSwissCompositeDataModule(
        dsconf.train_dataset,
        dsconf.valid_dataset,
        dsconf.datamodule,
    )

    model = LCNN(modelconf)

    # Callbacks
    model_ckpt = ModelCheckpoint(
        dirpath=f"checkpoints/{modelconf.train_params.savefile}",
        save_top_k=3,
        monitor="val_loss",
        save_on_train_epoch_end=False,
    )
    nowcast_image_logger = LogDiffNowcast(config=lognowcastconf.log_nowcast)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stopping = EarlyStopping(**modelconf.train_params.early_stopping)
    device_monitor = DeviceStatsMonitor()
    tb_logger = pl_loggers.TensorBoardLogger("/scratch/jritvane/src/swiss_data_test/logs", name=f"train_lcnn_{configpath}")
    profiler = PyTorchProfiler(profile_memory=False)

    trainer = pl.Trainer(
        strategy="ddp",
        num_nodes=1,
        profiler=profiler,
        logger=tb_logger,
        log_every_n_steps=1,
        val_check_interval=modelconf.train_params.val_check_interval,
        max_epochs=modelconf.train_params.max_epochs,
        max_time=modelconf.train_params.max_time,
        devices=modelconf.train_params.gpus,
        limit_val_batches=modelconf.train_params.val_batches,
        limit_train_batches=modelconf.train_params.train_batches,
        callbacks=[
            early_stopping,
            model_ckpt,
            lr_monitor,
            nowcast_image_logger,
            device_monitor
        ],
    )

    trainer.fit(model=model, datamodule=datamodel, ckpt_path=checkpoint)

    torch.save(model.state_dict(), f"checkpoints/{modelconf.training.savefile}/state_dict_{modelconf.train_params.savefile}.ckpt")
    trainer.save_checkpoint(f"checkpoints/{modelconf.training.savefile}/{modelconf.train_params.savefile}.ckpt")


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
