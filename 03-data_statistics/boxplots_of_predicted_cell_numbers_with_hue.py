import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Load a table of data
path_to_table1 = os.path.join('..', '..', '..', 'Notizen', 'Analyse Zellanzahl Vergleich der Trainings', 'Vorhersagen durch Segmentierung', 'box_plot.csv')
path_to_table1 = os.path.join('Anhang_B-D.csv')
predictions1 = pd.read_csv(path_to_table1, sep=';')

# Make a Boxplot
fig = plt.figure()
fig.autolayout : True
plt.tight_layout()
#ax = sns.boxplot(x=predictions1['rel_diff'], y=predictions1['T'], hue=predictions1['Net'])
ax = sns.boxplot(x=predictions1['Relative Differenz'], y=predictions1['Kultivierungszeitraum'], hue=predictions1['Netz'])
#ax.set(xlabel='Relative Abweichung der Kolokalisation mittels vorhergesagter Segmentierung \nvon der Kolokalisation mit der Ground-Truth-Segmentierung in %', ylabel='Kultivierungszeitraum')
ax.set(xlabel='Relative Abweichung der vorhergesagten Zellkernanzahl von der Ground-Truth in %', ylabel='Kultivierungszeitraum')
plt.savefig('box_plot.svg', format='svg', bbox_inches='tight')

# Make a Boxplot
plt.figure()
ax = sns.boxplot(x=predictions1['Percentual difference'], y=predictions1['Cultivation-period'], hue=predictions1['Netz'])
ax.set(xlabel='Relative Abweichung der Vorhersage von der Ground-Truth', ylabel='Kultivierungszeitraum')
plt.savefig('box_plot.svg', format='svg')
