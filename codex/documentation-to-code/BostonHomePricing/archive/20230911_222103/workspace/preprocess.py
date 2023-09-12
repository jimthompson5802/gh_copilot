import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Fill missing values with the median of the respective column
    df.fillna(df.median(), inplace=True)

    # Handle outliers using IQR method
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Feature Scaling
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df
