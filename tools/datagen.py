import numpy as np
import keras
import os
import nrrd
import sys
sys.path.append("..")
from tools import image_processing as impro
from tools import datatools

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, path_to_dataset, filenames_list, dim, channels=1, batch_size=32, 
                 shuffle=True, standardization_mode='volume_wise', linear_output_scaling_factor=1, 
                 border=None):
        
        # Initialize the class
        self.path_to_dataset = path_to_dataset
        self.filenames_list = filenames_list 
        self.channels = channels
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.standardization_mode = standardization_mode
        self.linear_output_scaling_factor = linear_output_scaling_factor
        self.border = border
    
    def on_epoch_end(self):
        #print('CALL on_epoch_end')
        
        # Shuffle indices after each epoch if shuffle == True
        self.indices = np.arange(len(self.filenames_list))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

            
    def __data_generation(self, filenames):
        # Load a batch of data
        X, y = datatools.load_data(path_to_dataset=self.path_to_dataset, 
                                   data_list=filenames, input_shape=self.dim, 
                                   standardization_mode=self.standardization_mode,
                                   border=self.border)
        
#        if self.standardization_mode != None:
#            standardize = True
#            #print('Datagen performing standardization...')
#        else:
#            standardize = False
#            #print('Datagen without standardization...')
#        X, y = datatools.load_data2(path_to_dataset=self.path_to_dataset, 
#                                   data_list=filenames, input_shape=self.dim, 
#                                   standardize=standardize,
#                                   border=self.border)
            
        # Scale the data
        y = y*self.linear_output_scaling_factor
        
        # Expand the dimension for channels
        X = X[:,:,:,:,np.newaxis]
        y = y[:,:,:,:,np.newaxis]
        
        return X, y
    

    def __len__(self):
        #print('CALL len')
        
        # Returns the number of batches per epoch
        return int(np.floor(len(self.filenames_list) / self.batch_size))
    
    
    def __getitem__(self, index):
        #print ('CALL getitem')
        self.on_epoch_end()
        # Generates a batch of data
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        files = [self.filenames_list[k] for k in indices]
        
        X, y = self.__data_generation(files)
        
        return X, y
    
        

