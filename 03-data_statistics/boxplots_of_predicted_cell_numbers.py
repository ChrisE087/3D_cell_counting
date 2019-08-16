import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Load a table of data
path_to_table = os.path.join('..', '..', '..', 'Notizen', 'Analyse Zellanzahl Vergleich der Trainings', 'predicted_cell_numbers_2019-08-09_12-07-12_100000.0_3_train_samples_fiji.csv')
predictions = pd.read_csv(path_to_table, sep=';')
predictions.reset_index()

# Print the name of the columns
print(predictions.columns)

# Make a Boxplot
plt.figure()
ax = sns.boxplot(x=predictions['Percentual difference'], y=predictions['Cultivation-period'])
ax.set(xlabel='Prozentuale Abweichung der Vorhersage von der Ground-Truth', ylabel='Typ')

# Make a Violine-plot
plt.figure()
ax = sns.violinplot(x=predictions['Percentual difference'], y=predictions['Cultivation-period'])
ax.set(xlabel='Prozentuale Abweichung der Vorhersage von der Ground-Truth', ylabel='Typ')
