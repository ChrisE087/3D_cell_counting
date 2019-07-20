import nrrd
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cc3d
import nrrd

def label_binary_image(data):
    

path_to_seg = os.path.join('..', '..', '..', 'Daten', '24h', 'untreated', 'C1-untreated_1.1_OpenSegSPIMResults_', 'Nucleisegmentedfill2r.nrrd')
data, header = nrrd.read(path_to_seg)

data = np.transpose(data, axes=(2,1,0)) #ZYX

labels = cc3d.connected_components(data, connectivity=6)

print(np.max(labels))

nrrd.write('labels.nrrd', header=header, data=np.transpose(labels, axes=(2,1,0)))

