import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
import pickle
import os
from preprocess_data import preprocess_data
from evaluate_correlations import evaluate_correlations

# Load data
data = pd.read_csv('../../generate-synthetic-data/workspace/data/housing2.csv')

# Preprocess data
data = preprocess_data(data)

# Evaluate correlations and remove low-correlation features
data = evaluate_correlations(data, 'MEDV')

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('MEDV', axis=1), data['MEDV'], test_size=0.2, random_state=42)

# Train models
models = {
    'LinearRegression': LinearRegression(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet()
}
for name, model in models.items():
    model.fit(X_train, y_train)

# Evaluate models
for name, model in models.items():
    predictions = model.predict(X_test)
    loss = mean_squared_error(y_test, predictions)
    print(f'{name} Loss: {loss}')

# Save models
for name, model in models.items():
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
