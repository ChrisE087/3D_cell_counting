import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Load a table of data
path_to_table = os.path.join('..', '..', '..', 'Notizen', 'Analyse Zellanzahl Vergleich der Trainings', 'Vorhersagen durch Segmentierung', '00.csv')
predictions = pd.read_csv(path_to_table, sep=';')
predictions.reset_index()

# Print the name of the columns
print(predictions.columns)

# Make a Boxplot
plt.figure()
ax = sns.boxplot(x=predictions['Percentual difference'], y=predictions['Cultivation-period'])
ax.set(xlabel='Relative Abweichung der Vorhersage von der Ground-Truth', ylabel='Typ')
plt.savefig('box_plot.svg', format='svg', bbox_inches='tight') # bbox_inches='tight' fixes the issue when a the axis labeling is cut off

# Make a Violine-plot
plt.figure()
ax = sns.violinplot(x=predictions['Percentual difference'], y=predictions['Cultivation-period'])
ax.set(xlabel='Relative Abweichung der Vorhersage von der Ground-Truth', ylabel='Typ', bbox_inches='tight')

plt.savefig('violine_plot.svg', format='svg')