import nrrd
import numpy as np
import os

path_to_data = os.path.join('..', '..', '..', 'Daten2', 'Fibroblasten', '20190221_1547_Segmentation', '1_draq5-NucleiBinary.nrrd') 
data, header = nrrd.read(path_to_data)

print(np.min(data))
print(np.max(data))

data[data > 0] = 1

print(np.min(data))
print(np.max(data))
