import nrrd
import os
import numpy as np
import matplotlib.pyplot as plt

path_to_data = os.path.join('..', '..', '..', 'Daten', '72h', 'untreated', 'C2-untreated_4_OpenSegSPIMResults_', 'gauss_centroids_opensegspim_seg.nrrd')
data, header = nrrd.read(path_to_data)
plt.figure()
plt.title('Verteilung der Werte in der Density-Map')
plt.xlabel('Intensitätswert der Density-Map vor Skalierung')
plt.ylabel('Anzahl der Werte')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
plt.hist(data.flatten(), bins=50)

data_scaled = np.copy(data)
data_scaled = data_scaled * 1e5
plt.figure()
plt.title('Verteilung der Werte in der Density-Map')
plt.xlabel('Intensitätswert der Density-Map nach Skalierung')
plt.ylabel('Anzahl der Werte')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
plt.hist(data_scaled.flatten(), bins=50)
