import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import DataLoader
from dataset import CustomDataset
from neural_network import NeuralNetwork

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader)}")

def calculate_loss(model, data_loader, criterion):
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
