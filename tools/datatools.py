import numpy as np
import os
import nrrd
import sys
sys.path.append("..")
from tools import image_processing as impro


def train_val_test_split(data_list, train_split, val_split, test_split, shuffle=True):
    # Shuffle the dataset
    if shuffle == True:
        np.random.shuffle(data_list)
    
    split_sum = train_split + val_split + test_split
    if split_sum > 1. or split_sum <= 0.:
        print("ERROR: train_split, val_split and test_split must be in range 0 to 1!")
        return
    
    # Calculate the list beginnings and endings of the list slices
    train_begin = 0
    train_end = int(np.ceil(train_split*len(data_list)))  
    val_begin = train_end
    val_end = val_begin+int(np.ceil(val_split*len(data_list)))
    test_begin = val_end
    test_end = test_begin+int(np.ceil(test_split*len(data_list)))
    
    # Split the dataset
    train = data_list[train_begin:train_end]
    val = data_list[val_begin:val_end]
    test = data_list[test_begin:test_end]
    
    return train, val, test

#def split_cultivation_period(path_to_dataset):
#    # Save all files in a list
#    files = os.listdir(path_to_dataset)
#    
#    # Split the files into cultivation period
#    data_list_24h = []
#    data_list_48h = []
#    data_list_72h = []
#    
#    for element in filter(lambda element: '24h' in element, files):
#        data_list_24h.append(element)
#        
#    for element in filter(lambda element: '48h' in element, files):
#        data_list_48h.append(element)
#        
#    for element in filter(lambda element: '72h' in element, files):
#        data_list_72h.append(element)
#        
#    return data_list_24h, data_list_48h, data_list_72h
    
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

def load_data(path_to_dataset, data_list, input_shape, 
              standardization_mode=None, border=None):

    # Make the data matrix
    X_data = np.zeros(shape=(np.size(data_list), input_shape[0], input_shape[1], 
                             input_shape[2]), dtype='float32')
    if border == None:
        y_data = np.zeros(shape=(np.size(data_list), input_shape[0], input_shape[1], 
                                 input_shape[2]), dtype='float32')
    else:
        y_data = np.zeros(shape=(np.size(data_list), input_shape[0]-2*border[0], 
                                 input_shape[1]-2*border[1], input_shape[2]-2*border[2]), dtype='float32')
    
    # Load the data into the data matrix
    for i in range(np.size(data_list)):
        filepath = os.path.join(path_to_dataset, data_list[i])
        data, header = nrrd.read(filepath)
        X_data[i,] = data[0,]
        if border == None:
            y_data[i,] = data[1,]
        else:
            y_data[i,] = impro.get_inner_slice(data[1,], border)
        
    # Standardize the input-data
    if standardization_mode == 'per_slice' or \
    standardization_mode == 'per_sample' or \
    standardization_mode == 'per_batch':
        X_data = impro.standardize_data(data=X_data, mode=standardization_mode)

    return X_data, y_data
            

def load_data2(path_to_dataset, data_list, input_shape, 
              standardize=False, border=None):

    # Make the data matrix
    X_data = np.zeros(shape=(np.size(data_list), input_shape[0], input_shape[1], 
                             input_shape[2]), dtype='float32')
    if border == None:
        y_data = np.zeros(shape=(np.size(data_list), input_shape[0], input_shape[1], 
                                 input_shape[2]), dtype='float32')
    else:
        y_data = np.zeros(shape=(np.size(data_list), input_shape[0]-2*border[0], 
                                 input_shape[1]-2*border[1], input_shape[2]-2*border[2]), dtype='float32')
    
    # Load the data into the data matrix
    for i in range(np.size(data_list)):
        filepath = os.path.join(path_to_dataset, data_list[i])
        data, header = nrrd.read(filepath)
        X_data[i,] = data[0,]
        if border == None:
            y_data[i,] = data[1,]
        else:
            y_data[i,] = impro.get_inner_slice(data[1,], border)
        
    # Standardize the input-data
    if standardize == True:
        X_data = impro.standardize_3d_images(X_data)

    return X_data, y_data

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
    
def get_balanced_dataset(path_to_dataset, clip=5000):
    # Clip is the maximal number of samples per range of patches with a number of cells in that range 
    range_dirs = get_immediate_subdirectories(path_to_dataset)
    dataset = []
    for i in range(len(range_dirs)):
        range_dir = os.path.join(path_to_dataset, range_dirs[i])
        files = [name for name in os.listdir(range_dir) if os.path.isfile(os.path.join(range_dir, name))]
        files = [os.path.join(range_dirs[i], file) for file in files] # Add the subdir
        if len(files) > clip:
            np.random.shuffle(files)
            dataset = dataset + files[0:clip]
            #print('Number of files in ', range_dir, ': ', len(files))
        else:
            dataset = dataset+files
    return dataset