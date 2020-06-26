import torch
from torch.utils.data import Dataset
import pandas as pd

class NNDataset(Dataset):
    def __init__(self, x, y):
        if isinstance(x.index, pd.MultiIndex):
            self.len = len(x.index.get_level_values('Date').unique())
            self.x = x.values.reshape(self.len, len(x.index.levels[-1]), -1)
        else:
            self.len = x.shape[0]
            self.x = x.values

        self.y = y.values

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
