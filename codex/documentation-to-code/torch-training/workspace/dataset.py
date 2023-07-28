import torch
from torch.utils.data import Dataset

class RegressionDataset(Dataset):
    def __init__(self, data):
        """
        Initialize the RegressionDataset.

        Args:
            data (pandas.DataFrame): Input data.
        """
        self.features = torch.tensor(data.drop('target', axis=1).values, dtype=torch.float32)
        self.target = torch.tensor(data['target'].values, dtype=torch.float32)

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Tuple containing the input features and target variable.
        """
        return self.features[idx], self.target[idx]
