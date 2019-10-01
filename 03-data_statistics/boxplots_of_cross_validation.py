import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
#from matplotlib import rc
#rc('text', usetex=True)

# Load a table of data
path_to_table1 = path_to_table1 = os.path.join('..', '..', '..', 'Ergebnisse', 'Kreuzvalidierung', 'Mittlerer Modellfehler', 'KV.csv')
predictions1 = pd.read_csv(path_to_table1, sep=';')

# Make a Boxplot
fig = plt.figure()
#fig.autolayout : True
#plt.tight_layout()
#ax = sns.boxplot(x=predictions1['rel_diff'], y=predictions1['T'], hue=predictions1['Net'])
ax = sns.boxplot(x=predictions1['Modellfehler'], y=predictions1['Komplexitaet'], hue=predictions1['Modellparameter'])
ax.set(xlabel=r'$\overline{E}$ auf den Testdaten', ylabel='Modellkomplexität')
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('box_plot.svg', format='svg', bbox_inches='tight')
