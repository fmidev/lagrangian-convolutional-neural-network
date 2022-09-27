"""Datamodule for FMI radar composite."""
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import LagrangianFMIComposite


class LagrangianFMICompositeDataModule(pl.LightningDataModule):
    def __init__(self, dsconfig, train_params, predict_list="predict"):
        super().__init__()
        self.dsconfig = dsconfig
        self.train_params = train_params
        self.predict_list = predict_list

    def prepare_data(self):
        # called only on 1 GPU
        pass

    def setup(self, stage):
        # called on every GPU
        self.train_dataset = LagrangianFMIComposite(split="train", **self.dsconfig.fmi)
        self.valid_dataset = LagrangianFMIComposite(split="valid", **self.dsconfig.fmi)
        self.test_dataset = LagrangianFMIComposite(split="test", **self.dsconfig.fmi)
        self.predict_dataset = LagrangianFMIComposite(
            split=self.predict_list, predicting=True, **self.dsconfig.fmi
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_params.train_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=_collate_fn,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.train_params.valid_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=_collate_fn,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.train_params.test_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=False,
            collate_fn=_collate_fn,
        )
        return test_loader

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.predict_dataset,
            batch_size=self.train_params.predict_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=False,
            collate_fn=_collate_fn,
        )
        return predict_loader


def _collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
