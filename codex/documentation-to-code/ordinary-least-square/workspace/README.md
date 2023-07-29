Based on the requirements, assumptions, and steps involved in building the OLS regression model, the following core classes, functions, and methods will be necessary:

1. `OLSModel` class:
   - `fit(X, y)`: Method to fit the OLS model to the given dataset.
   - `predict(X)`: Method to predict new values using the trained OLS model.
   - `evaluate(X, y)`: Method to evaluate the model's performance using MSE and R-squared metrics.
   - `visualize(X, y)`: Method to visualize the original data and the best-fitting line of the OLS model.

2. `mean_squared_error` function: Function to calculate the mean squared error between predicted and actual values.
3. `r_squared` function: Function to calculate the R-squared metric for the model's performance.
4. `plot_data` function: Function to plot the original data points and the best-fitting line of the OLS model.

Now, let's proceed with creating the necessary files and implementing the code.

**File 1: ols_model.py**

