import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Fill missing values with median
    data.fillna(data.median(), inplace=True)

    # Handle outliers using IQR method
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Feature scaling
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data
