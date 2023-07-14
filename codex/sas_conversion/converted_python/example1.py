import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

# Read data
data = pd.read_csv('Setosa.csv')

# Impute missing values
imputer = Imputer(missing_values='.', strategy='mean', axis=0)
imputer = imputer.fit(data.iloc[:, 2:4])
data.iloc[:, 2:4] = imputer.transform(data.iloc[:, 2:4])

# Plot correlation matrix
sns.set(style="white")
corr = data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Fisher (1936) Iris Setosa Data')
plt.show()