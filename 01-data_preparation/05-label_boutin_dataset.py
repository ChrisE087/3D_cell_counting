import nrrd
import numpy as np
import matplotlib.pyplot as plt
import os
import cc3d
import sys

sys.path.append("..")

path_to_dataset = os.path.join('..', '..', '..', 'Daten', 'Boutin', 'dataset')

files = os.listdir(path_to_dataset)
for i in range(len(files)):
    if 'Y' in files[i]:
        file = os.path.join(path_to_dataset, files[i])
        data, header = nrrd.read(file) #XYZ
        data = np.transpose(data, axes=(2,1,0)) #ZYX
        data = data.astype(np.uint8)
        labelled_data = cc3d.connected_components(data, connectivity=6)
        labelled_data = np.transpose(labelled_data, axes=(2,1,0)) #XYZ
        filename = os.path.splitext(files[i])[0]
        print('Cell number in ', filename, ': ', np.max(labelled_data))
        export_file = os.path.join(path_to_dataset, filename+'_labelled.nrrd')
        nrrd.write(filename=export_file, data=labelled_data, header=header)