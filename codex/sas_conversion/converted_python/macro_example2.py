import pandas as pd
from sklearn.linear_model import LinearRegression

def regression_macro(files):
    regression_results = pd.DataFrame()

    for file in files:
        data = pd.read_csv(file)
        X = data[['x1', 'x2', 'x3', 'x4']]
        y = data['y']

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        results = pd.DataFrame({'y_pred': y_pred})
        regression_results = pd.concat([regression_results, results])

    print(regression_results)

# Example usage
regression_macro(['file1.csv', 'file2.csv', 'file3.csv'])