# Commentary: add required imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import pandas as pd

# add import for TuneGridSearchCV
from tune_sklearn import TuneSearchCV

# add import for sp_randint
from scipy.stats import randint as sp_randint   # Commentary: as sp_randint is a more descriptive name


# read csv file into a pandas dataframe
df = pd.read_csv('data.csv')

# partition the data into training and testing sets
train, test = train_test_split(df, test_size=0.2)

# Train Random Forest model
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)

# run Ray Tune hyperparameter search for random forest model
tune_search = TuneSearchCV(  # Commentary: encountering error re: deprecated key word parameters
    rf,
    param_distributions={
        "max_depth": [3, None],
        "max_features": sp_randint(1, 11),
        "min_samples_split": sp_randint(2, 11),
        "bootstrap": [True, False],
        "criterion": ["mse", "mae"],
    },
    n_iter=10,
    random_state=0,
    n_jobs=-1,
    verbose=1,
    search_optimization="random",
)


