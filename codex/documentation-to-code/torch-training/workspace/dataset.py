import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.features = data.drop(columns=['target']).values
        self.target = data['target'].values

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.target[idx], dtype=torch.float32)
        return x, y
