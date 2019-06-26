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

def split_cultivation_period(path_to_dataset):
    # Save all files in a list
    files = os.listdir(path_to_dataset)
    
    # Split the files into cultivation period
    data_list_24h = []
    data_list_48h = []
    data_list_72h = []
    
    for element in filter(lambda element: '24h' in element, files):
        data_list_24h.append(element)
        
    for element in filter(lambda element: '48h' in element, files):
        data_list_48h.append(element)
        
    for element in filter(lambda element: '72h' in element, files):
        data_list_72h.append(element)
        
    return data_list_24h, data_list_48h, data_list_72h

def load_data(path_to_dataset, data_list, input_shape, standardize_input_data=False, 
              standardization_mode=None, border=None):
    # Get the shape of the data
#    filepath = os.path.join(path_to_dataset, data_list[0])
#    data, header = nrrd.read(filepath)
#    shape = data.shape
    
    # Make the data matrix
    X_data = np.zeros(shape=(np.size(data_list), input_shape[0], input_shape[1], 
                             input_shape[2]), dtype='float32')
    if border == None:
        y_data = np.zeros(shape=(np.size(data_list), input_shape[0], input_shape[1], 
                                 input_shape[2]), dtype='float32')
    else:
        y_data = np.zeros(shape=(np.size(data_list), input_shape[0]-2*border[0], 
                                 input_shape[1]-2*border[1], input_shape[2]-2*border[2]), dtype='float32')
    
    for i in range(np.size(data_list)):
        filepath = os.path.join(path_to_dataset, data_list[i])
        data, header = nrrd.read(filepath)
        X_data[i,] = data[0,]
        
        if standardize_input_data == True and standardization_mode == 'volume_wise':
            X_data[i,], mean, sigma = impro.standardize_data(X_data[i,])
        if border == None:
            y_data[i,] = data[1,]
        else:
            y_data[i,] = impro.get_inner_slice(data[1,], border)
    if standardize_input_data == True and standardization_mode == 'slice_wise':
        X_data = impro.standardize_dataset(input_dataset = X_data, mode='slice_wise')
        
    return X_data, y_data
            
            