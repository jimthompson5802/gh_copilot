import os
import pandas as pd

def save_dataset_to_csv(data, filepath):
    # Create the data folder if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save the dataset to a CSV file
    data.to_csv(filepath, index=False)
