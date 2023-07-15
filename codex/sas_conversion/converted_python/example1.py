import pandas as pd
import seaborn as sns

# Data on Iris Setosa
data = {
    'SepalLength': [50, 46, 46, 50, 48, 50, 43, 58, 51, 50, 57, 54, 50, 49, 54, 48, 45, 57, 51, 54, 51, 52, 53],
    'SepalWidth': [33, 34, 36, 35, 30, 30, 30, 40, 35, 34, 44, 39, 35, 31, 39, 30, 23, 38, 38, 34, 35, 37, 37],
    'PetalLength': [14, 14, None, 16, 14, 16, 11, 12, 14, 16, 15, None, 13, 15, 13, 14, 13, 17, 15, 15, 14, 15, 15],
    'PetalWidth': [2, 3, None, 6, 3, 2, None, 2, 2, 4, 4, 2, 2, 2, 4, 2, 3, 4, 2, 2, 2, 2, 2]
}

df = pd.DataFrame(data)

# Rename columns
df = df.rename(columns={
    'SepalLength': 'sepallength',
    'SepalWidth': 'sepalwidth',
    'PetalLength': 'petallength',
    'PetalWidth': 'petalwidth'
})

# Plot correlation matrix
sns.pairplot(df[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']])