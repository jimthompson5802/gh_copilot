# fix missing LudwigModel import error
from ludwig.api import LudwigModel

# define pandas, numpy
import pandas as pd
import numpy as np


# from local csv file, read in data
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

# define Ludwig config dictionary
config = {
    # define 3 numerical input features
    'input_features': [
        {'name': 'input_feat_1', 'type': 'numerical'},
        {'name': 'input_feat_2', 'type': 'numerical'},
        {'name': 'input_feat_3', 'type': 'numerical'},
    ],
    # define 1 binary output feature
    'output_features': [
        {'name': 'target', 'type': 'binary'},
    ],
    # train for 50 epochs, batch size of 64 and learning rate of 0.01
    'training': {'epochs': 50, 'batch_size': 64, 'learning_rate': 0.01},
}

# defin Ludwig model
model = LudwigModel(config)

# train model
train_stats = model.train(data_df=df)

