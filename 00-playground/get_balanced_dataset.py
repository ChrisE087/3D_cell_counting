import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import os
import nrrd
import matplotlib.pyplot as plt 

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
    
def get_balanced_dataset(path_to_dataset, clip=5000):
    # Clip is the maximal number of samples per range of patches with a number of cells in that range 
    range_dirs = get_immediate_subdirectories(path_to_dataset)
    dataset = []
    for i in range(len(range_dirs)):
        range_dir = os.path.join(path_to_dataset, range_dirs[i])
        # List all files
        files = [name for name in os.listdir(range_dir) if os.path.isfile(os.path.join(range_dir, name))]
        # Add the subdir to the file list
        files = [os.path.join(range_dirs[i], file) for file in files]
        if len(files) > clip:
            np.random.shuffle(files)
            files = files[0:clip]
            print(print('Number of files in ', range_dir, 'clipped to: ', len(files)))
            dataset = dataset + files
            #print('Number of files in ', range_dir, ': ', len(files))
        else:
            dataset = dataset+files
    return dataset

def split_cultivation_period(data_list):
    # Split the files into cultivation period
    data_list_24h = []
    data_list_48h = []
    data_list_72h = []
    
    for element in filter(lambda element: '24h' in element, data_list):
        data_list_24h.append(element)
        
    for element in filter(lambda element: '48h' in element, data_list):
        data_list_48h.append(element)
        
    for element in filter(lambda element: '72h' in element, data_list):
        data_list_72h.append(element)
        
    return data_list_24h, data_list_48h, data_list_72h

 
path_to_dataset = os.path.join('..', '..', '..', 'Daten', 'dataset_size32_stride16_split')
dataset = get_balanced_dataset(path_to_dataset=path_to_dataset, clip=5000)

# Generate a histogram to check the balanced dataset
cell_numbers = []
for i in range(len(dataset)):
    data, header = nrrd.read(os.path.join(path_to_dataset, dataset[i]))
    cell_numbers.append(np.sum(data[1,]))
    
# Plot the histogram
fig = plt.figure()
plt.title('Distribution of cell numbers (0-25 cells)')
plt.hist(cell_numbers, range=(0,5), bins=50)

# Split to cultivation period
data_list_24h, data_list_48h, data_list_72h = split_cultivation_period(dataset)
