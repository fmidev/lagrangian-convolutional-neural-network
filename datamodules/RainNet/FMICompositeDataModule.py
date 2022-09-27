"""Datamodule for FMI radar composite."""
import h5py
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import Rainnet_FMIComposite


class FMICompositeDataModule(pl.LightningDataModule):
    def __init__(self, dsconfig, train_params):
        super().__init__()
        self.dsconfig = dsconfig
        self.train_params = train_params

        if self.dsconfig.fmi.importer == "hdf5":
            self.db_path = dsconfig.db_path
            self.db = h5py.File(self.db_path, 'r')
        else:
            self.db_path = None
            self.db = None
        

    def prepare_data(self):
        # called only on 1 GPU
        pass

    def setup(self, stage):
        # called on every GPU
        self.train_dataset = Rainnet_FMIComposite(split="train", db=self.db,  **self.dsconfig.fmi)
        self.valid_dataset = Rainnet_FMIComposite(split="valid", db=self.db, **self.dsconfig.fmi)
        self.test_dataset = Rainnet_FMIComposite(split="test", db=self.db, **self.dsconfig.fmi)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_params.train_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.train_params.valid_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.train_params.test_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=False,
        )
        return test_loader
