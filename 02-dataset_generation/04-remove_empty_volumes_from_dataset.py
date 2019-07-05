import numpy as np
import os
import nrrd
import shutil
import sys
sys.path.append("..")

# Specify the thresholds
thresh_max_intensity = 70 # Maximum voxel intensity
thresh_cell_num = 0.1 # Number of cells in patch

path_to_dataset = os.path.join('..', '..', '..', 'Daten', 'dataset')
path_to_padding_set = os.path.join(path_to_dataset, 'padded_files')

# Create the target folder if it does not exist
if not os.path.exists(path_to_dataset):
    print('not exists')
    try:
        os.makedirs(path_to_padding_set)
    except OSError as e:
        if e.errno != e.errno.EEXIST:
            raise

file_list = os.listdir(path_to_dataset)
num_of_files = len(file_list)
file_cnt = 0
for filename in file_list:
    filepath = os.path.join(path_to_dataset, filename)
    try:
        data, header = nrrd.read(filepath)
    except PermissionError as e:
        print('Cannot open ', filename, ' permission error')
    
    if file_cnt % 100 == 0:
        progress = file_cnt*100/num_of_files
        sys.stdout.write('\r' + "{0:.2f}".format(progress) + '%')
        sys.stdout.flush()
    
    # Check if the input-data consists only of noise (maximum value is smaller
    # than thresh_max_intensity) or check if the file has too less cells
    # (sum over voxels must be greater than thresh_cell_num)
    
    max_intensity = np.max(data[0,])
    cell_num = np.sum(data[1,])
    
    if max_intensity <= thresh_max_intensity or cell_num <= thresh_cell_num:
        #print('Moving file ', filename, ' cell number: ', cell_num)
        shutil.move(filepath, path_to_padding_set)
    file_cnt = file_cnt + 1
        
