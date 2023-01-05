# Commentary: added required imports
from torch import nn, functional as F
from torch.utils.data import DataLoader, Dataset


# class for custom pytorch model
class CustomModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# class for custom pytorch dataset
class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


# create dataloader for custom dataset
def create_dataloader(data, target, batch_size):
    dataset = CustomDataset(data, target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader



