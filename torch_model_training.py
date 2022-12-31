# Commentary: manually added imports
import torch
from torch import nn, optim
from torch.nn import functional as F

import pandas as pd

# pytorch import Dataset and DataLoader
from torch.utils.data import Dataset, DataLoader


# define torch nn.module with 1 input layer, 3 hidden layers, and 1 output layer
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Commentary: added comment to specify input layer
        self.input = nn.Linear(5, 10)   # Commentary: changed input layer to 5 input features to match data set
        self.hidden1 = nn.Linear(10, 10)
        self.hidden2 = nn.Linear(10, 10)
        self.hidden3 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu_(self.input(x))
        x = F.relu_(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.output(x)
        return x

# define function to train model
def train_model(model, train_loader, num_epochs=100):
    # define loss function
    criterion = nn.MSELoss()
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # iterate through epochs
    for epoch in range(num_epochs):
        # iterate through batches
        for i, data in enumerate(train_loader):
            # get inputs and labels
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            outputs = model(inputs)
            # calculate loss
            loss = criterion(outputs, labels)
            # backward pass
            loss.backward()
            # update weights
            optimizer.step()
        # print loss every 5 epochs
        if (epoch+1) % 5 == 0:  # Commentary: modified from batches to epochs
            print('Epoch: {}/{}, Step: {}/{}, Loss: {}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

# define function to test model
def test_model(model, test_loader):
    with torch.no_grad():
        correct = 0
        total = 0
        criterion = nn.MSELoss()  # Commentary: added criterion to fix unresolved reference
        # iterate through test data
        for data in test_loader:
            # get inputs and labels
            inputs, labels = data
            # forward pass
            outputs = model(inputs)
            # calculate loss
            loss = criterion(outputs, labels)
            # calculate accuracy
            total += labels.size(0)
            correct += (outputs == labels).sum().item()
        # print accuracy
        print('Accuracy: {}%'.format(100 * correct / total))

# define function to save model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# define data loader for csv file
class CSVDataset(Dataset):
    def __init__(self, path):
        # read csv file
        self.data = pd.read_csv(path)
        # get number of columns
        self.len = self.data.shape[0]
        # get input features
        self.x_data = torch.from_numpy(self.data.iloc[:, 0:-1].values).float()
        # get target features
        self.y_data = torch.from_numpy(self.data.iloc[:, -1].values).view(-1, 1).float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

# define function to generate data loader
def get_loader(path, batch_size):
    dataset = CSVDataset(path)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return loader

# define function to generate model
def get_model():
    model = Net()
    return model

# function to train model
def main():
    # define path for csv file
    path = 'data.csv'
    # define batch size
    batch_size = 32
    # define number of epochs
    num_epochs = 100
    # define path to save model
    save_path = 'model.pth'
    # get data loader
    train_loader = get_loader(path, batch_size)
    # get model
    model = get_model()
    # train model
    train_model(model, train_loader, num_epochs)
    # save model
    save_model(model, save_path)

if __name__ == '__main__':
    main()
