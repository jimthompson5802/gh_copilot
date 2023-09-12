Based on the requirements, we will need the following core classes, functions, and methods:

1. pandas.read_csv: To read the CSV file into a pandas DataFrame.
2. pandas.DataFrame.fillna: To fill missing values with the median of the respective column.
3. pandas.DataFrame.quantile: To calculate the IQR for handling outliers.
4. sklearn.preprocessing.StandardScaler: To perform feature scaling.
5. sklearn.model_selection.train_test_split: To split the data into training and testing sets.
6. sklearn.linear_model.LinearRegression, sklearn.linear_model.Lasso, sklearn.linear_model.ElasticNet: To create the regression models.
7. sklearn.metrics.mean_squared_error: To calculate the loss value (MSE).
8. joblib.dump: To save the models.

Now, let's write the code. We will put everything in a single Python script named `housing_regression.py`.

housing_regression.py
