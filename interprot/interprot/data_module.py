import multiprocessing

import polars as pr
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from utils import train_val_test_split
from torch.utils.data import DataLoader, Dataset


class PolarsDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        return {"Sequence": row["Sequence"], "Entry": row["Entry"]}


# Data Module
class SequenceDataModule(pl.LightningDataModule):
    def __init__(self, data_path=None, batch_size=48, num_workers=0, eve_data=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        df = pr.read_parquet(self.data_path)
        dataset = PolarsDataset(df)

        self.train_data, self.val_data, self.test_data = train_val_test_split(dataset)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
