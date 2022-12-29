

# create a synthetic regression dataset with 5 features and 1000  samples and create a pandas dataframe
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np

# create a synthetic regression dataset with 5 features and 1000  samples and create a pandas dataframe
X, y = make_regression(n_samples=1000, n_features=5, n_informative=5, random_state=1)
df = pd.DataFrame(X, columns=['input_feat_1', 'input_feat_2', 'input_feat_3', 'input_feat_4', 'input_feat_5'])
df['target'] = y

# print first 5 rows of dataframe
print(df.head())   # Commentary: wrapped print statement

# save dataframe to csv file
df.to_csv('data.csv', index=False)
