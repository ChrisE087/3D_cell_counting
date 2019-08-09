import seaborn as sns
import pandas as pd
import os

# Plot from a sample dataset
sns.set(style='whitegrid')
flights = sns.load_dataset('flights')
ax = sns.boxplot(x=flights['year'], y=flights['passengers'])

# Plot from own dataset
path_to_table = os.path.join('test_data', 'predicted_cell_numbers_2019-08-09_12-07-12_100000.0_3_train_samples_fiji.csv')
predictions = pd.read_csv(path_to_table, sep=';')
predictions.reset_index()
print(predictions.columns)
sns.boxplot(x=predictions['percdiff'])
ax = sns.boxplot(x=predictions['percdiff'], y=predictions['cultivation_period'])
sns.violinplot(x=predictions['percdiff'], y=predictions['cultivation_period'])
