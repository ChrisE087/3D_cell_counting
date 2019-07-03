import os
import numpy as np
from shutil import copyfile


import_path = os.path.join('..', '..', '..', 'Daten', 'dataset')
export_path = os.path.join('..', '..', '..', 'Daten', 'dataset_mini')

p_train = 0.025
p_test = 0.001
p_val = 0.001

# Load the files into list
dataset_list = os.listdir(import_path)

# Shuffle the list
np.random.shuffle(dataset_list)

# Extract the train, test and validation files
train_start = 0
train_end = int(p_train*np.size(dataset_list))
val_start = train_end + 1
val_end = val_start + int(p_val*np.size(dataset_list))
test_start = val_end + 1
test_end = test_start + int(p_test*np.size(dataset_list))
train_list = dataset_list[train_start:train_end]
val_list = dataset_list[val_start:val_end]
test_list = dataset_list[test_start:test_end]

for i in range(np.size(train_list)):
    src = os.path.join(import_path, train_list[i])
    dst = os.path.join(export_path, 'train', train_list[i])
    copyfile(src, dst)
    
for i in range(np.size(val_list)):
    src = os.path.join(import_path, val_list[i])
    dst = os.path.join(export_path, 'val', val_list[i])
    copyfile(src, dst)
    
for i in range(np.size(test_list)):
    src = os.path.join(import_path, test_list[i])
    dst = os.path.join(export_path, 'test', test_list[i])
    copyfile(src, dst)
    
