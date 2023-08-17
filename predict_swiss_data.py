"""This script will run nowcasting prediction for the L-CNN model implementation
"""
import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from utils import load_config, setup_logging
from utils import LagrangianHDF5Writer
from models import LCNN
from datasets import LagrangianSwissCompositeDataModule, LagrangianSwissCompositeDataset


def run(checkpointpath, configpath, predict_list) -> None:

    confpath = Path("config") / configpath
    dsconf = load_config(confpath / "lagrangian_datasets.yaml")
    outputconf = load_config(confpath / "output.yaml")
    modelconf = load_config(confpath / "lcnn.yaml")

    setup_logging(outputconf.logging)

    # dataset = LagrangianSwissCompositeDataset(**dsconf.valid_dataset, split="predict")
    # for batch in dataset:
    #     datewindow = dataset.get_window(batch["idx"])
    #     common_time = dataset.get_common_time(batch["idx"])

    #     import ipdb

    #     ipdb.set_trace()

    #     print(batch)

    datamodel = LagrangianSwissCompositeDataModule(
        dsconf.train_dataset,
        dsconf.valid_dataset,
        dsconf.datamodule,
        predict_list=predict_list,
    )

    model = LCNN(modelconf).load_from_checkpoint(checkpointpath, config=modelconf, map_location=torch.device('cpu'))

    output_writer = LagrangianHDF5Writer(**modelconf.prediction_output)

    # tb_logger = pl_loggers.TensorBoardLogger("logs", name=f"predict_{configpath}")
    trainer = pl.Trainer(
        strategy="ddp",
        num_nodes=1,
        devices=1,
        profiler="pytorch",
        # logger=tb_logger,
        callbacks=[output_writer],
    )

    # Predictions are written in HDF5 file
    trainer.predict(model, datamodel, return_predictions=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    argparser.add_argument(
        "config",
        type=str,
        help="Configuration folder path",
    )
    argparser.add_argument(
        "-l",
        "--list",
        type=str,
        default="predict",
        help="Name of predicted list (replaces {split} in dataset settings).",
    )

    args = argparser.parse_args()

    run(args.checkpoint, args.config, args.list)
