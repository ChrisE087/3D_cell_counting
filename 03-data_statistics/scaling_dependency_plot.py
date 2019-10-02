import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

path_to_table = path_to_table1 = os.path.join('..', '..', '..', 'Ergebnisse', 'Skalierungsabhaengigkeit', 'Skalierungsabhaengigkeit.csv')
errors = pd.read_csv(path_to_table, sep=';')

fig, ax = plt.subplots()
plt.plot([0.,1.3], [0,0], color='grey', linewidth=1)
sns.lineplot(x=errors['Skalierungsfaktor'], y=errors['Relativer Fehler'], hue=errors['Sphaeroid'])
#plt.xlabel('')
#plt.xticks(rotation=30, ha='right')
plt.legend(bbox_to_anchor=(0.61, 0.95), loc='best', borderaxespad=0.)
plt.ylabel(r'$Err_{rel}$')
ax.legend().texts[0].set_text('Sphäroid')
#plt.legend(title='Modellparameter', bbox_to_anchor=(0.5, 1.25), loc='upper center', borderaxespad=0.)
plt.savefig('line_plot.svg', format='svg', bbox_inches='tight')