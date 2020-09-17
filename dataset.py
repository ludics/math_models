import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class MyDataset(Dataset):
    def __init__(self, kind="positive"):
        self.data_dir = "datas/train_data/" + kind
        self.data_paths = [os.path.join(self.data_dir, afile) for afile in os.listdir(self.data_dir)]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, i):
        with open(self.data_paths[i], "rb") as infile:
            data = np.load(infile) # len x 20

        data = (data-data.mean(axis=0)) / data.std(axis=0)

        return {"data": torch.from_numpy(data).permute(1,0)}






