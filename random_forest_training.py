import pandas as pd

# read csv file into pandas dataframe
df = pd.read_csv('data.csv')

# print first 5 rows of dataframe
df.head()

# print last 5 rows of dataframe
df.tail()

# Random forest regression model on the dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# split data into training and testing sets
train, test = train_test_split(df, test_size=0.2)

# separate the target variable from the training and testing sets
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = test.drop(['target'], axis=1)
test_y = test['target']

# create a random forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# fit the regressor with x and y data
rf.fit(train_x, train_y)

# make predictions on the test set
pred = rf.predict(test_x)

# calculate the mean squared error
mse = mean_squared_error(test_y, pred)
print('Mean Squared Error:', mse)

# save the model to disk
import pickle

filename = 'finalized_model.sav'
pickle.dump(rf, open(filename, 'wb'))

