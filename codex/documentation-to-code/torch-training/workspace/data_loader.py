import pandas as pd

def load_data(file_path):
    """
    Load data from the synthetic_regression.csv file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded data.
    """
    data = pd.read_csv(file_path)
    return data
