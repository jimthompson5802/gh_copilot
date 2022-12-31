# Commentary: add required imports
import pandas as pd
import matplotlib.pyplot as plt


# read csv file into a pandas dataframe
df = pd.read_csv('data.csv')

# function to input the dataframe and perform bootstrap analysis on 100 samples
# for a specified column in the dataframe and return the mean and standard deviation for each bootstrap sample
def bootstrap(df, column):
    # create empty lists to store the mean and standard deviation for each bootstrap sample
    means = []
    stds = []
    # loop through 100 samples
    for i in range(100):
        # create a bootstrap sample of the dataframe
        bootstrap_sample = df.sample(frac=1, replace=True)
        # calculate the mean and standard deviation of the bootstrap sample
        means.append(bootstrap_sample[column].mean())
        stds.append(bootstrap_sample[column].std())
    # return the mean and standard deviation for each bootstrap sample
    return means, stds

# plot historgram the mean and standard deviation for each bootstrap sample and save the plot in the plots folder
def plot(means, stds):
    # create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # plot the mean for each bootstrap sample
    ax1.hist(means, bins=10)
    ax1.set_title('Mean')  # Commentary: added this call
    # plot the standard deviation for each bootstrap sample
    ax2.hist(stds, bins=10)
    ax2.set_title('Standard Deviation')  # Commentary: add this call
    # save the plot in the plots folder
    plt.savefig('plots/bootstrap_plot.png')  # Commentary: updated file name

# setup main
def main():
    # call bootstrap function to get the mean and standard deviation for each bootstrap sample
    means, stds = bootstrap(df, 'input_feat_1')  # Commentary: specified column
    # call plot function to plot the mean and standard deviation for each bootstrap sample
    plot(means, stds)

# call main
if __name__ == '__main__':
    main()



