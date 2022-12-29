import pandas as pd

from ludwig.api import LudwigModel

# read csv file into pandas dataframe
df = pd.read_csv('data.csv')

# define Ludwig configuration
config = {
    'input_features': [
        # numerical feature called my_feature
        {'name': 'my_feature', 'type': 'numerical'},
        # categorical feature called my_category
        {'name': 'my_category', 'type': 'category'},
        # image feature called my_image with torchvision encoder resnet-18
        {'name': 'my_image', 'type': 'image', 'encoder': 'resnet-18'}  # Commentary: requires modification
    ],
    'output_features': [
        # categorical feature called target
        {'name': 'target', 'type': 'category'}
    ],
    # train for 50 epochs with learning rate 0.0002
    'training': {'epochs': 50, 'learning_rate': 0.0002}
}

# create a Ludwig model
model = LudwigModel(config)

# train the model
train_stats = model.train(data_df=df)

# make predictions on the test set
predictions = model.predict(data_df=df)



