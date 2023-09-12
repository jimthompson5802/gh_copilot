import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle
import os
from preprocess import preprocess_data
from correlations import evaluate_correlations

def load_data(filepath):
    # Load data from CSV file into a pandas DataFrame
    df = pd.read_csv(filepath)
    return df

def split_data(df, target, test_size=0.2):
    # Split the data into a training set and a test set
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    # Train multiple regression models
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

def evaluate_models(models, X_test, y_test):
    # Evaluate the models and print the loss value
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{name} MSE: {mse}")

def save_models(models, directory="models"):
    # Save the trained models to disk
    if not os.path.exists(directory):
        os.makedirs(directory)
    for name, model in models.items():
        filepath = os.path.join(directory, f"{name.replace(' ', '_').lower()}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(model, f)

def main():
    # Load data
    df = load_data("../../generate-synthetic-data/workspace/data/housing2.csv")

    # Preprocess data
    df = preprocess_data(df)

    # Evaluate correlations and remove low-correlation features
    df = evaluate_correlations(df, "MEDV")

    # Split data
    X_train, X_test, y_train, y_test = split_data(df, "MEDV")

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate models
    evaluate_models(models, X_test, y_test)

    # Save models
    save_models(models)

if __name__ == "__main__":
    main()
