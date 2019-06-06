import numpy as np
import matplotlib.pyplot as plt
import nrrd
import keras
import os
import sys
sys.path.append("..")
from tools.cnn import CNN
from tools import datagen
from tools import image_processing as impro
from tools import datatools

###############################################################################
# Define the parameters
###############################################################################

# Dataset Parameters
path_to_dataset = os.path.join('..', '..', 'Daten', 'dataset')
train_split = 0.05
val_split = 0.05
test_split = 0.05

# Model Parameters
input_shape=(32, 32, 32, 1)
#hidden_layer_activation='relu'
hidden_layer_activation = keras.layers.LeakyReLU(alpha=0.2)
output_layer_activation = None
padding = 'same'

# Data Generator parameters
channels = input_shape[-1]
shuffle = True
normalize_input_data = False
standardize_input_data = True

# Training parameters
learning_rate = 0.00001
epochs = 16
batch_size = 64
optimizer = keras.optimizers.adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, 
                                  epsilon=None, decay=0.0, amsgrad=False)
loss = 'mse'
metrics = ['mae']
tensor_board_cb = keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=0, 
                                              write_graph=True, write_images=True)
callbacks = [tensor_board_cb]

###############################################################################
# Prepare the training data
###############################################################################

# Make a list of training files
data_list = os.listdir(path_to_dataset)

# Split the file list of the dataset into spheroid cultivation period
data_list_24h, data_list_48h, data_list_72h = datatools.split_cultivation_period(path_to_dataset)

# Split the list of training files into train validation and test data
train_24h, val_24h, test_24h = datatools.train_val_test_split(data_list_24h, train_split, 
                                                            val_split, test_split, shuffle=True)
train_48h, val_48h, test_48h = datatools.train_val_test_split(data_list_48h, train_split, 
                                                            val_split, test_split, shuffle=True)
train_72h, val_72h, test_72h = datatools.train_val_test_split(data_list_72h, train_split, 
                                                            val_split, test_split, shuffle=True)

# Concatenate the list of training files
train_list = train_24h + train_48h + train_72h
val_list = val_24h + val_48h + val_72h
test_list = test_24h + test_48h + test_72h

test_list = np.sort(test_list)

###############################################################################
# Define the model
###############################################################################

cnn = CNN()
cnn.define_model(input_shape=input_shape, filters_exp=5, kernel_size=(3, 3, 3), 
                  pool_size=(2, 2, 2), hidden_layer_activation=hidden_layer_activation, 
                  output_layer_activation=output_layer_activation, padding=padding)

cnn.compile_model(loss=loss, optimizer=optimizer, metrics=metrics)

###############################################################################
# Fit the model
###############################################################################

train_generator = datagen.DataGenerator(path_to_dataset=path_to_dataset, 
                                        filenames_list=train_list, dim=input_shape[0:3], 
                                        channels=channels, batch_size=batch_size, shuffle=shuffle, 
                                        normalize_input_data=normalize_input_data, 
                                        standardize_input_data=standardize_input_data)
val_generator = datagen.DataGenerator(path_to_dataset=path_to_dataset, 
                                      filenames_list=val_list, dim=input_shape[0:3], 
                                      channels=channels, batch_size=batch_size, shuffle=shuffle, 
                                      normalize_input_data=normalize_input_data, 
                                      standardize_input_data=standardize_input_data)

history = cnn.fit_model(epochs=epochs, train_generator=train_generator, val_generator=val_generator, 
                       callbacks=callbacks)

###############################################################################
# Evaluate the model
###############################################################################


###############################################################################
# Predict some data
###############################################################################
X_test_data, y_test_data = datatools.load_data(path_to_dataset, test_list, 
                                               standardize_input_data=True)
X_test = X_test_data[1,]
y_test = y_test_data[1,]

#X_test = np.expand_dims(X_test, axis=3)
y_pred = cnn.predict_sample(X_test)
y_pred = y_pred[0,:,:,:,0]

plt.imshow(y_test[:,:,16])
plt.imshow(y_pred[:,:,16])
print('Number of cells (ground truth): ', np.sum(y_test))
print('Number of cells (predicted): ', np.sum(y_pred))
