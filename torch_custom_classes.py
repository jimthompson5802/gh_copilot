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


# Commentary: notice mis-spellings in the following comment
# pytroch custome model class MyClass with 4 layers of 256, 128, 64, 32 nodes
class MyClass(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyClass, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
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



