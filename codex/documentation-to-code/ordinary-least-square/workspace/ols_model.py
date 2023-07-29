import numpy as np
from metrics import mean_squared_error, r_squared
from visualization import plot_data

class OLSModel:
    def __init__(self):
        self.coefficients = None
    
    def fit(self, X, y):
        # Add a column of ones to X for the intercept term
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Calculate the coefficients using the OLS formula
        self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
    
    def predict(self, X):
        # Add a column of ones to X for the intercept term
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Predict the values using the trained model
        return X @ self.coefficients
    
    def evaluate(self, X, y):
        # Predict the values using the trained model
        y_pred = self.predict(X)
        
        # Calculate the mean squared error
        mse = mean_squared_error(y, y_pred)
        
        # Calculate the R-squared metric
        r2 = r_squared(y, y_pred)
        
        return mse, r2
    
    def visualize(self, X, y):
        # Plot the original data points
        plot_data(X, y)
        
        # Add a column of ones to X for the intercept term
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Predict the values using the trained model
        y_pred = X @ self.coefficients
        
        # Plot the best-fitting line of the OLS model
        plt.plot(X[:, 1], y_pred, color='red')
        plt.show()
