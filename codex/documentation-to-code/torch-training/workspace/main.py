import data_loader
import dataset
import model
import train

# Step 1: Load data
data = data_loader.load_data("../../generate-synthetic-data/workspace/data/synthetic_regression.csv")

# Step 2: Convert to PyTorch dataset
dataset = dataset.RegressionDataset(data)

# Step 3: Define neural network
input_size = len(data.columns) - 1
hidden_size = 64
output_size = 1
neural_network = model.NeuralNetwork(input_size, hidden_size, output_size)

# Step 4: Train the neural network
num_epochs = 10
learning_rate = 0.001
train.train_model(neural_network, dataset, num_epochs, learning_rate)

# Step 5: Save the model
model.save_model(neural_network, "trained_model.pt")
