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
                 shuffle=True, normalize_input_data=False, standardize_input_data=False, 
                 linear_output_scaling_factor=1):
        
        # Initialize the class
        self.path_to_dataset = path_to_dataset
        self.filenames_list = filenames_list 
        self.channels = channels
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize_input_data = normalize_input_data
        self.standardize_input_data = standardize_input_data
        self.linear_output_scaling_factor = linear_output_scaling_factor
    
    def on_epoch_end(self):
        #print('CALL on_epoch_end')
        
        # Shuffle indices after each epoch if shuffle == True
        self.indices = np.arange(len(self.filenames_list))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            
#    def __data_generation(self, filenames):
#        #print('CALL data_generation')
#        
#        # Generates a batch of data
#        X = np.empty((self.batch_size, *self.dim, self.channels), dtype=np.float32)
#        y = np.empty((self.batch_size, *self.dim, self.channels), dtype=np.float32)
#        
#        for i, filename in enumerate(filenames):
#            path = os.path.join(self.path_to_dataset, filename)
#            data, header = nrrd.read(path)
#            model_input = data[0,]
#            model_target = data[1,]
#            if self.normalize_input_data == True:
#                model_input, max_val, min_val = impro.normalize_data(model_input)
#            if self.standardize_input_data == True:
#                model_input, mean, sigma = impro.standardize_data(model_input)
#            X[i,:,:,:,0] = model_input
#            y[i,:,:,:,0] = model_target
#        
#        return X, y
            
    def __data_generation(self, filenames):
        # Generates a batch of data
        X, y = datatools.load_data(path_to_dataset=self.path_to_dataset, 
                                   data_list=filenames, input_shape=self.dim, 
                                   standardize_input_data=self.standardize_input_data)
        if self.normalize_input_data == True:
            print('WARNING: Datset-wise Data normalization in datagen not implemented yet')
        y = y*self.linear_output_scaling_factor
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
    
        

