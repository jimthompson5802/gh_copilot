import pandas as pd

def evaluate_correlations(data, target):
    # Calculate correlations
    correlations = data.corr()[target]

    # Remove low-correlation features
    data = data.drop(correlations[correlations.abs() < 0.5].index, axis=1)

    return data
