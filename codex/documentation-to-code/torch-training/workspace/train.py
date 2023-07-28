import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, dataset, num_epochs, learning_rate):
    """
    Train the neural network on the dataset.

    Args:
        model (torch.nn.Module): Neural network model.
        dataset (torch.utils.data.Dataset): Training dataset.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def save_model(model, file_path):
    """
    Save the trained model.

    Args:
        model (torch.nn.Module): Trained neural network model.
        file_path (str): Path to save the model.
    """
    torch.save(model.state_dict(), file_path)
