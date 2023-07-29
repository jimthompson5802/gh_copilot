from ols_model import OLSModel
import numpy as np

# Create a sample dataset
X = np.array([[1, 2], [3, 4], [15, 6]])
y = np.array([3, 5, 7])

# Create an instance of the OLSModel class
model = OLSModel()

# Fit the OLS model to the dataset
model.fit(X, y)

# Predict new values using the trained model
X_new = np.array([[7, 8], [9, 10]])
y_pred = model.predict(X_new)
print("Predicted values:", y_pred)

# Evaluate the model's performance
mse, r2 = model.evaluate(X, y)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Visualize the original data and the best-fitting line
model.visualize(X, y)
