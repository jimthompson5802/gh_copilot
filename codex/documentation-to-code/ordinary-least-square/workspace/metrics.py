import numpy as np

def mean_squared_error(y_true, y_pred):
    # Calculate the mean squared error
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def r_squared(y_true, y_pred):
    # Calculate the total sum of squares
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Calculate the residual sum of squares
    rss = np.sum((y_true - y_pred) ** 2)
    
    # Calculate the R-squared metric
    r2 = 1 - (rss / tss)
    return r2
