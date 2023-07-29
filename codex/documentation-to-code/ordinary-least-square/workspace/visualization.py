import matplotlib.pyplot as plt

def plot_data(X, y):
    # Plot the original data points
    plt.scatter(X, y)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Original Data')
    plt.show()
