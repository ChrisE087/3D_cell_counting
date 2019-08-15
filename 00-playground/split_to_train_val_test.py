import sys
sys.path.append("..")
import os
import numpy as np
import random
from tools import datatools

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
            train_spheroids = train_spheroids.get(spheroid_type)
            val_spheroids = val_spheroids.get(spheroid_type)
            test_spheroids = test_spheroids.get(spheroid_type)
            
            for j in range(len(train_spheroids)):
                spheroid = train_spheroids[j]
                if spheroid_type == 'untreated':
                    filename = spheroid
                else:
                    filename = spheroid_type+'_'+spheroid
                # Get all filenames of the patches belonging to the actual spheroid
                subset = get_patch_filenames(patches_files, spheroid_type, filename)
                dataset_table.append([spheroid_type, spheroid, 'training'])
                train_files.extend(subset)
            
            for j in range(len(val_spheroids)):
                spheroid = val_spheroids[j]
                if spheroid_type == 'untreated':
                    filename = spheroid
                else:
                    filename = spheroid_type+'_'+spheroid
                # Get all filenames of the patches belonging to the actual spheroid
                subset = get_patch_filenames(patches_files, spheroid_type, filename)
                dataset_table.append([spheroid_type, spheroid, 'validation'])
                val_files.extend(subset)
                
            for j in range(len(test_spheroids)):
                spheroid = test_spheroids[j]
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
                
#%% Testing the function

path_to_dataset = os.path.join('..', '..', '..', 'Datensaetze', 'dataset_density-maps_fiji_and_mathematica')

# Complete disctionary (Key=spheroid-type, value=list of spheroid names)
# of data that can be used for training, validating and testing
# Comment if you wish to not use it in the dataset

# Choose spheroids from the complete dataset
spheroid_names = {'untreated': ['24h_C2-untreated_1.1',
                                '24h_C2-untreated_1.2',
                                '24h_C2-untreated_2.1',
                                '24h_C2-untreated_2.2',
                                '24h_C2-untreated_2.3',
                                '24h_C2-untreated_3',
                                '48h_C2-untreated_1',
                                '48h_C2-untreated_2',
                                '48h_C2-untreated_3',
                                '48h_C2-untreated_4.1',
                                '72h_C2-untreated_1',
                                '72h_C2-untreated_2',
                                '72h_C2-untreated_3',
                                '72h_C2-untreated_4'],
                 'Fibroblasten': ['1_draq5',
                                  '2_draq5',
                                  '3_draq5',
                                  '4_draq5',
                                  '5_draq5',
                                  '7_draq5',
                                  '8_draq5',
                                  '9_draq5',
                                  '10_draq5'],
                 'Hacat': ['C3-2',
                          'C3-3',
                          'C3-4',
                          'C3-5',
                          'C3-6',
                          'C3-7',
                          'C3-8',
                          'C3-9'],
                 'HT29': ['C2-HT29_Glycerol_Ki67_01',
                          'C2-HT29_Glycerol_Ki67_02',
                          'C2-HT29_Glycerol_Ki67_03',
                          'C2-HT29_Glycerol_Ki67_03-1',
                          'C2-HT29_Glycerol_Ki67_04',
                          'C2-HT29_Glycerol_Ki67_05',
                          'C2-HT29_Glycerol_Ki67_07',
                          'C2-HT29_Glycerol_Ki67_09',
                          'C2-HT29_Glycerol_Ki67_10'],  
                 'HTC8': ['C3-2a',
                          'C3-3',
                          'C3-5',
                          'C3-6l',
                          'C3-6r',
                          'C3-8_',
                          'C3-8c',
                          'C3-9',
                          'C3-10'],
                 'NPC1': ['C3-2',
                          'C3-3',
                          'C3-4',
                          'C3-5',
                          'C3-6',
                          'C3-7',
                          'C3-8',
                          'C3-9']}

# Choose spheroids only from dataset 1
spheroid_names2 = {'untreated': ['24h_C2-untreated_1.1',
                                '24h_C2-untreated_1.2',
                                '24h_C2-untreated_2.1',
                                '24h_C2-untreated_2.2',
                                '24h_C2-untreated_2.3',
                                '24h_C2-untreated_3',
                                '48h_C2-untreated_1',
                                '48h_C2-untreated_2',
                                '48h_C2-untreated_3',
                                '48h_C2-untreated_4.1',
                                '72h_C2-untreated_1',
                                '72h_C2-untreated_2',
                                '72h_C2-untreated_3',
                                '72h_C2-untreated_4']}

# Example dictionary for choosing the spheroids of the dataset manually    
train_spheroids = {'untreated': ['24h_C2-untreated_1.1',
                                '24h_C2-untreated_1.2',
                                '48h_C2-untreated_1',
                                '48h_C2-untreated_2',
                                '72h_C2-untreated_1',
                                '72h_C2-untreated_2']}
    
val_spheroids = {'untreated': [ '24h_C2-untreated_2.1',
                                '48h_C2-untreated_3',
                                '72h_C2-untreated_3']}
    
test_spheroids = {'untreated': [ '24h_C2-untreated_2.2',
                                '48h_C2-untreated_4.1',
                                '72h_C2-untreated_4']}

# Specify the number of spheroids per dataset
num_train_spheroids = 4
num_val_spheroids = 1
num_test_spheroids = 1

train_files, val_files, test_files, dataset_table = get_datasets(path_to_dataset=path_to_dataset, spheroid_names=spheroid_names2, 
                                                                 num_train_spheroids=num_train_spheroids, num_val_spheroids=num_val_spheroids, 
                                                                 num_test_spheroids=num_test_spheroids, train_spheroids=train_spheroids, 
                                                                 val_spheroids=val_spheroids, test_spheroids=test_spheroids)