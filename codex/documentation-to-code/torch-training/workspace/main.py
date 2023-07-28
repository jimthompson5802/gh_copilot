import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from data_loader import DataLoader
from dataset import CustomDataset
from neural_network import NeuralNetwork
from train import train_model, calculate_loss, save_model

# Set the file path for the data
file_path = "../../generate-synthetic-data/workspace/data/synthetic_regression.csv"

# Set the hyperparameters
input_size = 10  # Assuming there are 10 input features
hidden_size = 100
output_size = 1  # Assuming the target variable is a single value
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Load the data
data_loader = DataLoader(file_path)
data = data_loader.load_data()

# Convert the data to a PyTorch dataset
dataset = CustomDataset(data)

# Split the dataset into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders for training and test sets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create the neural network model
model = NeuralNetwork(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Calculate the loss on the test set
test_loss = calculate_loss(model, test_loader, criterion)
print(f"Test Loss: {test_loss}")

# Save the trained model
save_model(model, "trained_model.pt")
