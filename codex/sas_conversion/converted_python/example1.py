import pandas as pd
import seaborn as sns

# Read data
data = pd.read_csv('Setosa.csv')

# Label columns
data.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

# Plot correlation matrix
sns.heatmap(data.corr(), annot=True)