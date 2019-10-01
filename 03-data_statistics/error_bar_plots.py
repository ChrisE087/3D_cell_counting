import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

path_to_table = path_to_table1 = os.path.join('..', '..', '..', 'Dokumente', 'Thesis', 'Tabellen', 'Mittlerer_Modellfehler_Seg-Net1B-2A.csv')
errors = pd.read_csv(path_to_table, sep=';')

fig, ax = plt.subplots(figsize=(4.0, 5.0))
sns.barplot(x=errors['Datensatz'], y=errors['Mittlerer Modellfehler'], hue=errors['Modellparameter'])
plt.ylabel(r'$\overline{E}$')
plt.xlabel('')
plt.xticks(rotation=30, ha='right')
#plt.legend(bbox_to_anchor=(0.61, 0.95), loc='best', borderaxespad=0.)
plt.legend(bbox_to_anchor=(0.7, 0.98), loc='best', borderaxespad=0.)
#plt.legend(title='Modellparameter', bbox_to_anchor=(0.5, 1.25), loc='upper center', borderaxespad=0.)
plt.savefig('bar_plot.svg', format='svg', bbox_inches='tight')