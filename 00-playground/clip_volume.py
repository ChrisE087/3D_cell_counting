import numpy as np

a = np.arange(-8192, 24576, 1)
a = np.reshape(a, (32, 32, 32))

print('Min: ', np.min(a))
print('Max: ', np.max(a))

a = np.clip(a, 0, np.max(a))

print('Min: ', np.min(a))
print('Max: ', np.max(a))

import nrrd
import os
path_to_nuclei = os.path.join('..', '..', '..', 'Daten', '24h', 'untreated', 'C1-untreated_1.1_OpenSegSPIMResults_', 'gauss_centroids.nrrd')
data, header = nrrd.read(path_to_nuclei)
print(np.sum(data))
print(np.min(data))
print(np.max(data))
