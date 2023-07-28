from csv_loader import load_dataset
from neural_network import NeuralNetwork, train_model, save_model

# Load the dataset
dataset = load_dataset("data.csv")

# Define the neural network architecture
input_size = len(dataset[0][0])
hidden_size = 64
output_size = 1
model = NeuralNetwork(input_size, hidden_size, output_size)

# Train the model
num_epochs = 10
learning_rate = 0.001
train_model(model, dataset, num_epochs, learning_rate)

# Save the model
save_model(model, "model.pth")
