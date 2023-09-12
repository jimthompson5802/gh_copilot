import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from joblib import dump

# Load data
df = pd.read_csv('../../generate-synthetic-data/workspace/data/housing2.csv')

# Preprocessing
# Fill missing values with median
df.fillna(df.median(), inplace=True)

# Handle outliers using IQR method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Feature Scaling
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Split data into features and target variable
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train models
models = {
    'Linear Regression': LinearRegression(),
    'Lasso Regression': Lasso(),
    'ElasticNet Regression': ElasticNet()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'{name} MSE: {mse}')

    # Save model
    dump(model, f'{name.replace(" ", "_").lower()}_model.joblib')
