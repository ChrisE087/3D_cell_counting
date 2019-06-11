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
from tools.cnn import CNN
from tools import datagen
from tools import image_processing as impro
from tools import datatools


'''
 ' Huber loss.
 ' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
 ' https://en.wikipedia.org/wiki/Huber_loss
'''
def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = tf.keras.backend.abs(error) < clip_delta

  squared_loss = 0.5 * tf.keras.backend.square(error)
  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

  return tf.where(cond, squared_loss, linear_loss)

'''
 ' Same as above but returns the mean loss.
'''
def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
  return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))

###############################################################################
# Define the parameters
###############################################################################

# Dataset Parameters
path_to_dataset = os.path.join('..', '..', '..', 'Daten', 'dataset')
train_split = 0.05
val_split = 0.001
test_split = 0.001
data_shape = (32, 32, 32)
channels = 1

# Model Parameters
input_shape = data_shape + (channels,)
hidden_layer_activation='relu'
#hidden_layer_activation = keras.layers.LeakyReLU(alpha=0.2)
output_layer_activation = None
padding = 'same'

# Data Generator parameters
shuffle = False
normalize_input_data = False
standardize_input_data = True
linear_output_scaling_factor = 2048000

# Training parameters
#learning_rate = 0.00001
learning_rate = 0.001
epochs = 32
batch_size = 64
optimizer = keras.optimizers.adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, 
                                  epsilon=None, decay=0.0, amsgrad=False)
loss = huber_loss
metrics = ['mae']

###############################################################################
# Define the model callbacks
###############################################################################
model_export_path = os.path.join('model_export')
checkpoint = ModelCheckpoint(filepath=os.path.join(model_export_path, 'weights.hdf5'), 
                             monitor='val_loss', mode='min', verbose=1, save_best_only=True)

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
logdir = os.path.join('logs', timestamp)
tensor_board_cb = TensorBoard(log_dir=logdir, histogram_freq=0, 
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
                                                            val_split, test_split, shuffle=shuffle)
train_48h, val_48h, test_48h = datatools.train_val_test_split(data_list_48h, train_split, 
                                                            val_split, test_split, shuffle=shuffle)
train_72h, val_72h, test_72h = datatools.train_val_test_split(data_list_72h, train_split, 
                                                            val_split, test_split, shuffle=shuffle)

# Concatenate the list of training files
train_list = train_24h + train_48h + train_72h
val_list = val_24h + val_48h + val_72h
test_list = test_24h + test_48h + test_72h

# Undo the shuffeling for testing
test_list = np.sort(test_list)
test_list = test_list.tolist()

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
                                        filenames_list=train_list, dim=data_shape, 
                                        channels=channels, batch_size=batch_size, shuffle=shuffle, 
                                        normalize_input_data=normalize_input_data, 
                                        standardize_input_data=standardize_input_data,
                                        linear_output_scaling_factor=linear_output_scaling_factor)
val_generator = datagen.DataGenerator(path_to_dataset=path_to_dataset, 
                                      filenames_list=val_list, dim=data_shape, 
                                      channels=channels, batch_size=batch_size, shuffle=shuffle, 
                                      normalize_input_data=normalize_input_data, 
                                      standardize_input_data=standardize_input_data,
                                      linear_output_scaling_factor=linear_output_scaling_factor)

history = cnn.fit_generator(epochs=epochs, train_generator=train_generator, val_generator=val_generator, 
                       callbacks=callbacks)

###############################################################################
# Evaluate the model
###############################################################################

X_test_data, y_test_data = datatools.load_data(path_to_dataset=path_to_dataset, 
                                               data_list=train_list, input_shape=data_shape,
                                               standardize_input_data=standardize_input_data)
test_loss = cnn.evaluate_model(X_test=np.expand_dims(X_test_data, axis=4), 
                   y_test=np.expand_dims(y_test_data, axis=4), batch_size=batch_size)

###############################################################################
# Predict some data
###############################################################################

X_test = X_test_data[55,]
y_test = y_test_data[55,]

#X_test = np.expand_dims(X_test, axis=3)
y_pred = cnn.predict_sample(X_test)
y_pred = y_pred[0,:,:,:,0]/linear_output_scaling_factor

plt.imshow(y_test[:,:,16])
plt.imshow(y_pred[:,:,16])
print('Number of cells (ground truth): ', np.sum(y_test))
print('Number of cells (predicted): ', np.sum(y_pred))
