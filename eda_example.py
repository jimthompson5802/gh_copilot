# Commentary: specify imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# remove plots folder if it exists and recreate it
# !rm -rf    # Commentary: commented out because generate command only work in jupyter notebook
# !mkdir plots # Commentary: commented out because generate command only work in jupyter notebook
import os
# if os.path.exists('plots'):  # Commentary: commented out because wrong command
#     os.rmdir('plots')
import shutil   # Commentary: manaully added
shutil.rmtree("plots", ignore_errors=True)  # Commentary: manaully added
os.mkdir('plots')

# create pandas dataframe from csv file
df = pd.read_csv('data.csv')

# Commentary: modfied comment to specify saving plot in plots folder
# plot input_feat_1 vs target feature and save to file in plots folder
plt.plot(df['input_feat_1'], df['target'])
plt.savefig('plots/plot.png')

# generate coorelation plot and save to file in plots folder
corr = df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, vmax=1, square=True, annot=True, cmap='cubehelix')
plt.savefig('plots/corr.png')

# generate pairplot and save to file in plots folder
sns.pairplot(df)
plt.savefig('plots/pairplot.png')

# generate scatterplot matrix and save to file in plots folder
pd.plotting.scatter_matrix(df, figsize=(10, 10))
plt.savefig('plots/scatterplot.png')

# generate boxplot and save to file in plots folder
df.boxplot()
plt.savefig('plots/boxplot.png')

# generate histogram and save to file in plots folder
df.hist()
plt.savefig('plots/histogram.png')
