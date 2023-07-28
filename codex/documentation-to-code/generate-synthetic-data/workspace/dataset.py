import numpy as np
import pandas as pd

def generate_synthetic_regression_dataset():
    # Set the random seed
    np.random.seed(0)

    # Generate the informative features
    informative_features = np.random.uniform(low=0, high=1, size=(100, 5))

    # Generate the redundant features
    redundant_features = np.random.uniform(low=0, high=1, size=(100, 5))

    # Generate the target variable with Gaussian noise
    coefficients = np.random.uniform(low=0, high=1, size=5)
    intercept = np.random.uniform(low=0, high=1)
    noise = np.random.normal(loc=0, scale=0.5, size=100)
    target = np.dot(informative_features, coefficients) + intercept + noise

    # Combine the features and target into a DataFrame
    features = np.concatenate((informative_features, redundant_features), axis=1)
    feature_names = [f'feature_{i+1}' for i in range(10)]
    data = pd.DataFrame(features, columns=feature_names)
    data['target'] = target

    return data
