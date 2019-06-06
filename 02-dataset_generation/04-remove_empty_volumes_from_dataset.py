import numpy as np
import os
import nrrd
import shutil
import sys
sys.path.append("..")

path_to_dataset = os.path.join('..', '..', 'Daten', 'dataset')
path_to_padding_set = os.path.join(path_to_dataset, 'padded_files')

# Create the target folder if it does not exist
if not os.path.exists(path_to_dataset):
    print('not exists')
    try:
        os.makedirs(path_to_padding_set)
    except OSError as e:
        if e.errno != e.errno.EEXIST:
            raise

for filename in os.listdir(path_to_dataset):
    filepath = os.path.join(path_to_dataset, filename)
    try:
        data, header = nrrd.read(filepath)
    except PermissionError as e:
        print('Cannot open ', filename, ' permission error')
    # Check if the file exclusively consists of padding (sum over voxels must 
    # be greater than threshold)
    threshold = 0.001    # Number of cells in patch
    if np.sum(data[1,]) < threshold:
        print('Moving file ', filename)
        shutil.move(filepath, path_to_padding_set)
        
