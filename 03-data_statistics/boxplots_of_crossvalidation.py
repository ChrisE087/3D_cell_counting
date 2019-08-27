import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Load a table of data
path_to_table1 = os.path.join('..', '04-conv_net', 'model_export', 'dataset1_cross_validation', 'density_maps', 'Test-Losses_after_128_epochs.csv')
path_to_table2 = os.path.join('..', '04-conv_net', 'model_export', 'dataset_mix_cross_validation', 'density_maps', 'Test-Losses_after_128_epochs.csv')
table1 = pd.read_csv(path_to_table1, sep=';')
table2 = pd.read_csv(path_to_table2, sep=';')

# Print the name of the columns
print(table1.columns)
print(table2.columns)

# Make a Boxplot
plt.figure()
ax = sns.boxplot(x=table1['MSE on test-data'])
ax = sns.boxplot(x=table2['MSE on test-data'])
ax.set(xlabel='MSE')

plt.figure()
ax = sns.boxplot(x=table1['MAE on test-data'])
ax = sns.boxplot(x=table2['MAE on test-data'])
ax.set(xlabel='MAE')
plt.show()

# Make a Violine-plot
# Make a Boxplot
plt.figure()
ax2 = sns.violinplot(x=table1['MSE on test-data'], palette='inferno')
ax1 = sns.violinplot(x=table2['MSE on test-data'])
ax1.set(xlabel='MSE')

plt.figure()
ax1 = sns.violinplot(x=table1['MAE on test-data'], palette='inferno')
ax2 = sns.violinplot(x=table2['MAE on test-data'])
ax1.set(xlabel='MAE')
plt.show()