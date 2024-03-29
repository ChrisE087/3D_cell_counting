import numpy as np
import os
import nrrd
import sys
import random
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
    
def split_cultivation_period(data_list, shuffle=True):
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
        
    if shuffle == True:
        np.random.shuffle(data_list_24h)
        np.random.shuffle(data_list_48h)
        np.random.shuffle(data_list_72h)
        
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

class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
    
def get_patch_filenames(all_patches_files, spheroid_type, spheroid_name):
    # Getting all filenames of the patches from spheroid_type->spheroid_name in a subset-list of files
    indices = [k for k, element in enumerate(all_patches_files) if spheroid_type in element and spheroid_name in element]
    return [all_patches_files[k] for k in indices]

def get_datasets(path_to_dataset, spheroid_names, num_train_spheroids=4, num_val_spheroids=1, num_test_spheroids=1, train_spheroids=None, val_spheroids=None, test_spheroids=None):
    patches_files = os.listdir(path_to_dataset)
    
    spheroid_types = list(spheroid_names.keys())
    
    total_spheroids = num_train_spheroids + num_val_spheroids + num_test_spheroids

    # Lists for the filenames of train, val and test patches
    train_files = []
    val_files = []
    test_files = []
    
    dataset_table = [['Type', 'Spheroid', 'purpose']]
    
    for i in range(len(spheroid_types)):
        
        spheroid_type = spheroid_types[i]
        print('--------Choosing train, val and test patches for ', spheroid_type)
        
        # Choose the spheroids from the given dictionarys
        if train_spheroids != None and val_spheroids != None and test_spheroids != None:
            train_spheroids_type = train_spheroids.get(spheroid_type)
            val_spheroids_type = val_spheroids.get(spheroid_type)
            test_spheroids_type = test_spheroids.get(spheroid_type)
            
            for j in range(len(train_spheroids_type)):
                spheroid = train_spheroids_type[j]
                if spheroid_type == 'untreated':
                    filename = spheroid
                else:
                    filename = spheroid_type+'_'+spheroid
                # Get all filenames of the patches belonging to the actual spheroid
                subset = get_patch_filenames(patches_files, spheroid_type, filename)
                dataset_table.append([spheroid_type, spheroid, 'training'])
                train_files.extend(subset)
            
            for j in range(len(val_spheroids_type)):
                spheroid = val_spheroids_type[j]
                if spheroid_type == 'untreated':
                    filename = spheroid
                else:
                    filename = spheroid_type+'_'+spheroid
                # Get all filenames of the patches belonging to the actual spheroid
                subset = get_patch_filenames(patches_files, spheroid_type, filename)
                dataset_table.append([spheroid_type, spheroid, 'validation'])
                val_files.extend(subset)
                
            for j in range(len(test_spheroids_type)):
                spheroid = test_spheroids_type[j]
                if spheroid_type == 'untreated':
                    filename = spheroid
                else:
                    filename = spheroid_type+'_'+spheroid
                # Get all filenames of the patches belonging to the actual spheroid
                subset = get_patch_filenames(patches_files, spheroid_type, filename)
                dataset_table.append([spheroid_type, spheroid, 'testing'])
                test_files.extend(subset)
            
        # Choose random spheroids
        else:
            # Get a list of spheroids for the actual type
            spheroid_list = spheroid_names.get(spheroid_type)
            
            # Choose n random elements from the list
            chosen_spheroids = random.sample(spheroid_list, total_spheroids)
            
            print('Dataset for ', spheroid_type, ' consists of:\n', chosen_spheroids)
            
            # Check if the training patches of these random chosen spheroids are in the dataset
            # If not, an exception is raised
            for j in range(len(chosen_spheroids)):
                spheroid = chosen_spheroids[j]
                
                if spheroid_type == 'untreated':
                    filename = spheroid
                else:
                    filename = spheroid_type+'_'+spheroid
                
                if not any(filename in s for s in patches_files):
                    #print(spheroid_type, '->', spheroid, ' is not in the dataset')
                    # Abort, when a specified spheroid is not part of the dataset
                    error_msg = spheroid_type+'->'+spheroid+' is not part of the dataset! Please check your dictionary of available spheroid-patches. Now aborting...'
                    print(error_msg)
                    raise Error(error_msg)
        
                # If no exception is raised, all patches of the chosen spheroids are in the dataset -> Fill the lists for the filenames of train, val and testpatches
                
                # Calculate the intervals for train, val and test
                start_idx_train = 0
                end_idx_train = num_train_spheroids
                start_idx_val = end_idx_train
                end_idx_val = start_idx_val + num_val_spheroids
                start_idx_test = end_idx_val
                end_idx_test = total_spheroids
                
                # Get all filenames of the patches belonging to the actual spheroid
                subset = get_patch_filenames(patches_files, spheroid_type, filename)
                
                if j >= start_idx_train and j < end_idx_train:
                    dataset_table.append([spheroid_type, spheroid, 'training'])
                    #print(spheroid, ' is used for train')
                    
                    # Extend the training list with the filenames of the actual spheroid
                    train_files.extend(subset)
                    
                if j >= start_idx_val and j < end_idx_val:
                    dataset_table.append([spheroid_type, spheroid, 'validation'])
                    #print(spheroid, ' is used for validataion')
                    
                    # Extend the validation list with the filenames of the actual spheroid
                    val_files.extend(subset)
                if j >= start_idx_test and j < end_idx_test:
                    dataset_table.append([spheroid_type, spheroid, 'testing'])
                    #print(spheroid, ' is used for testing')
                    
                    # Extend the test list with the filenames of the actual spheroid
                    test_files.extend(subset)
    
    return train_files, val_files, test_files, dataset_table