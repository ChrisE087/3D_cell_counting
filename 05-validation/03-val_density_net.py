import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import keras
import os
import time
import datetime
import nrrd
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tools.cnn import CNN
from tools.cnn import dice_coef_loss
from tools import datagen
from tools import image_processing as impro
from tools import datatools
import pickle
import pandas as pd
import json

model_import_path = os.path.join('..', '04-conv_net', 'model_export', 'CROSS-VALIDATION', 'complexity_16', '2019-09-06_00-18-55_100000.0')
weights = 'best_weights' #'model_weights' or 'best_weights'

###############################################################################
# Define the parameters
###############################################################################
# Dictionary for choosing random spheroids for training out of the following dataset
# Comment single spheroids if you want to use a subset
spheroid_names_dataset_all = {'untreated': ['24h_C2-untreated_1.1',
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
                          'C2-HT29_Glycerol_Ki67_03_',
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
                 
# Choose if you want to choose spheroids only from dataset 1
spheroid_names_dataset_1 = {'untreated': ['24h_C2-untreated_1.1',
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
    
#test_spheroids = {'untreated': [ '24h_C2-untreated_2.1',
#                                 '24h_C2-untreated_2.2']}
#test_spheroids = {'untreated': [ '24h_C2-untreated_2.2'],
#                  'Fibroblasten': ['4_draq5'],
#                  'Hacat': ['C3-5'],
#                  'HT29': ['C2-HT29_Glycerol_Ki67_09'],
#                  'HTC8': ['C3-8_'],
#                  'NPC1': ['C3-9']}

test_spheroids = {'untreated': ['72h_C2-untreated_1'],
                  'Fibroblasten': ['8_draq5'],
                  'Hacat': ['C3-7'],
                  'HT29': ['C2-HT29_Glycerol_Ki67_05'],
                  'HTC8': ['C3-9'],
                  'NPC1': ['C3-7']}
    
# Specify the number of spheroids per dataset
num_train_spheroids = 4
num_val_spheroids = 1
num_test_spheroids = 1


# Dataset Parameters
path_to_dataset = os.path.join('..', '..', '..', 'Datensaetze', 'dataset_density-maps_fiji_and_mathematica_filtered')
data_shape = (32, 32, 32)
channels = 1

# Model Parameters
input_shape = data_shape + (channels,)

# Data Generator parameters
train_shuffle = True
val_shuffle = False
standardization_mode = 'per_sample' # 'per_slice', 'per_sample' or 'per_batch'
border = None # None or border in each dimension around the inner slice which should be extracted
linear_output_scaling_factor = 1e5

# Training parameters
learning_rate = 0.005
batch_size = 256
optimizer = keras.optimizers.adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, 
                                  epsilon=None, decay=0.0, amsgrad=False)
#optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=0., decay=0., nesterov=False)
evaluate = True

#optimizer = keras.optimizers.SGD(lr=learning_rate)
loss = ['MSE']
metrics = ['MAE']

###############################################################################
# Prepare the training data
###############################################################################
# Example if you want to choose the spheroids used for train val and testing randomized
train_files, val_files, test_files, dataset_table = datatools.get_datasets(path_to_dataset=path_to_dataset, spheroid_names=spheroid_names_dataset_all, 
                                                                 num_train_spheroids=num_train_spheroids, num_val_spheroids=num_val_spheroids, 
                                                                 num_test_spheroids=num_test_spheroids, train_spheroids=test_spheroids, 
                                                                 val_spheroids=test_spheroids, test_spheroids=test_spheroids)

# Load unstandardized test data
X_test_data, y_test_data = datatools.load_data(path_to_dataset=path_to_dataset, 
                                               data_list=test_files, input_shape=data_shape,
                                               standardization_mode=None,
                                               border=border)

###############################################################################
# Load the model
###############################################################################
#cnn.load_model_single_file(import_path, 'model_single')
cnn = CNN(linear_output_scaling_factor=linear_output_scaling_factor, 
          standardization_mode=standardization_mode)
cnn.load_model_json(model_import_path, 'model_json', weights)

# Evaluate the model on the test data
summary = cnn.compile_model(loss=loss, optimizer=optimizer, metrics=metrics)
test_loss_best_weights = cnn.evaluate_model(X_test=X_test_data, y_test=y_test_data, batch_size=batch_size)
print(test_loss_best_weights)

# Export the value of the test-loss
test_loss_export_path = os.path.join('test_loss_best_weights.txt')
with open(test_loss_export_path,'w') as file:
    for l in range(len(test_loss_best_weights)):
        file.write(str(test_loss_best_weights[l])+'\n')