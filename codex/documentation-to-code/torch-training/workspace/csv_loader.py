import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.transform = ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data.iloc[idx, :-1].values.astype(float)
        y = self.data.iloc[idx, -1]
        return self.transform(x), y


def load_dataset(file_path):
    dataset = CustomDataset(file_path)
    return dataset
