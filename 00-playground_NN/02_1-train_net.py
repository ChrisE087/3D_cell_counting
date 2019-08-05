import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import keras
import os
import time
import datetime
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.models import Model, load_model, model_from_json
from keras.layers import Input, Conv3D, Conv3DTranspose, ReLU, LeakyReLU, Activation, MaxPooling3D, UpSampling3D, Cropping3D, BatchNormalization
from keras.layers.merge import concatenate
from keras import backend as K
from keras.regularizers import l2
from tools.cnn import CNN
from tools import datagen
from tools import image_processing as impro
from tools import datatools

def conv3d_block(input_tensor, n_filters, kernel_size=3, padding='same', 
                     activation='relu', batchnorm=True):
        norm_axis = -1
        # First Conv->Activation->Batch-Norm layer
        x = Conv3D(filters=n_filters, kernel_size=(kernel_size, kernel_size, kernel_size), padding=padding,
                   activation=None, use_bias=False, kernel_initializer='glorot_uniform')(input_tensor)
        if activation == 'leaky_relu':
            x = LeakyReLU(alpha=0.3)(x)
        else:
            x = Activation('relu')(x)
        if batchnorm:
            x = BatchNormalization(axis=norm_axis)(x)
        # Second Conv->Activation->Batch-Norm layer
        x = Conv3D(filters=n_filters, kernel_size=(kernel_size, kernel_size, kernel_size), padding=padding,
                   activation=None, use_bias=False, kernel_initializer='glorot_uniform')(x)
        if activation == 'leaky_relu':
            x = LeakyReLU(alpha=0.3)(x)
        else:
            x = Activation('relu')(x)
        if batchnorm:
            x = BatchNormalization(axis=norm_axis)(x)
        return x

#%% Load the data    
linear_output_scaling_factor = 409600000000
path_to_dataset = os.path.join('..', '..', '..', 'Daten', 'dataset_size32_stride16_split')
data_list = datatools.get_balanced_dataset(path_to_dataset=path_to_dataset, clip=5000)

# Shuffle the dataset
np.random.shuffle(data_list)

train_list = data_list[0:10000]
val_list = data_list[10000:15000]
test_list = data_list[15000:200000]

X_train, y_train = datatools.load_data(path_to_dataset=path_to_dataset, 
                                       data_list=train_list, 
                                       input_shape=(32,32,32), 
                                       standardization_mode='per_sample', 
                                       border=None)
X_val, y_val = datatools.load_data(path_to_dataset=path_to_dataset, 
                                       data_list=val_list, 
                                       input_shape=(32,32,32), 
                                       standardization_mode='per_sample', 
                                       border=None)
X_test, y_test = datatools.load_data(path_to_dataset=path_to_dataset, 
                                       data_list=test_list, 
                                       input_shape=(32,32,32), 
                                       standardization_mode=None, 
                                       border=None)

# Expand the dimensions for channels
X_train = X_train[:,:,:,:, np.newaxis]
y_train = y_train[:,:,:,:, np.newaxis]
y_train = y_train * linear_output_scaling_factor
X_val = X_val[:,:,:,:, np.newaxis]
y_val = y_val[:,:,:,:, np.newaxis]
y_val = y_val * linear_output_scaling_factor
#%% Define the model
n_filters = 16
kernel_size = 3
pool_size = 2
hidden_layer_activation = 'relu'
output_layer_activation = None
batchnorm = True
padding = 'same'
input_shape = (32,32,32,1)

inputs = Input(input_shape)
ce1 = conv3d_block(inputs, n_filters=n_filters*1, kernel_size=kernel_size, 
                          activation=hidden_layer_activation, batchnorm=batchnorm)
pe1 = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size), strides=None, 
                             padding=padding)(ce1)
ce2 = conv3d_block(pe1, n_filters=n_filters*2, kernel_size=kernel_size, 
                          activation=hidden_layer_activation, batchnorm=batchnorm)
pe2 = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size), strides=None, 
                             padding=padding)(ce2)
ce3 = conv3d_block(pe2, n_filters=n_filters*4, kernel_size=kernel_size, 
                          activation=hidden_layer_activation, batchnorm=batchnorm)
ud1 = UpSampling3D(size=(pool_size, pool_size, pool_size))(ce3)
cd1 = conv3d_block(ud1, n_filters=n_filters*2, kernel_size=kernel_size, 
                          activation=hidden_layer_activation, batchnorm=batchnorm)
ud2 = UpSampling3D(size=(pool_size, pool_size, pool_size))(cd1)
cd2 = conv3d_block(ud2, n_filters=n_filters*1, kernel_size=kernel_size, 
                          activation=hidden_layer_activation, batchnorm=batchnorm)
outputs = Conv3D(filters=1, kernel_size=(1, 1, 1), 
                        activation=output_layer_activation)(cd2)
model = Model(inputs=[inputs], outputs=[outputs])

#%% Fit the model
opt = keras.optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss=['mse'], optimizer=opt, metrics=['mae'])
model.fit(X_train, y_train, batch_size=64, epochs=16)