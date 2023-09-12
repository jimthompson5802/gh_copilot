def evaluate_correlations(df, target):
    # Evaluate the correlations between each feature and the target feature
    correlations = df.corr()[target]

    # Remove features with low correlation
    low_correlation_features = correlations[correlations.abs() < 0.5].index
    df = df.drop(low_correlation_features, axis=1)

    return df
