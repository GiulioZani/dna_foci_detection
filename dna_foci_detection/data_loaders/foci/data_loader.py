import pytorch_lightning as pl
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import h5py
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    Dataset,
    WeightedRandomSampler,
)
from .dataset import FociDataset
import pandas as pd
import numpy as np
import os
import ipdb
import torch as t


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.data_location = params.data_location
        self.train_batch_size = params.train_batch_size
        self.test_batch_size = params.test_batch_size

        num_workers = os.cpu_count()
        self.num_workers = num_workers
        with h5py.File(params.data_location, "r") as h5data:
            filenames = np.array(
                [file[:-6] for file in list(h5data.keys()) if "image" in file]
            )
        permuted_indices = t.randperm(len(filenames))
        valid_size = int(len(filenames) * 0.1)
        self.val_filenames = filenames[permuted_indices[:valid_size]]
        self.test_filenames = filenames[permuted_indices[:valid_size]]
        self.train_filenames = filenames[permuted_indices[2 * valid_size :]]
        self.out_len = params.out_len

    def train_dataloader(self):
        dataset = FociDataset(
            hdf5_filename=self.data_location,
            filenames=self.train_filenames,
            split="train",
            crop_size=self.params.crop_size,
            out_len=self.out_len
        )
        return DataLoader(
            dataset, batch_size=self.train_batch_size, shuffle=True
        )

    def val_dataloader(self):
        dataset = FociDataset(
            hdf5_filename=self.data_location,
            filenames=self.val_filenames,
            split="val",
            crop_size=self.params.crop_size,
            out_len=self.out_len
        )
        return DataLoader(
            dataset, batch_size=self.test_batch_size, shuffle=True
        )

    def test_dataloader(self):
        dataset = FociDataset(
            hdf5_filename=self.data_location,
            filenames=self.val_filenames,
            split="test",
            crop_size=self.params.crop_size,
            out_len=self.out_len
        )
        return DataLoader(
            dataset, batch_size=self.test_batch_size, shuffle=True
        )
