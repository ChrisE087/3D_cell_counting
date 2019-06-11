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
num_of_train_samples = 500
train_split = 0.8
val_split = 0.1
test_split = 0.1
data_shape = (32, 32, 32)
channels = 1
data_scaling_factor = 2048000

# Model Parameters
input_shape = data_shape + (channels,)
hidden_layer_activation='relu'
output_layer_activation = None
padding = 'same'

# Data Generator parameters
shuffle = False
normalize_input_data = False
standardize_input_data = True

# Training parameters
#learning_rate = 0.00001
learning_rate = 0.001
epochs = 32
batch_size = 64
optimizer = keras.optimizers.adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, 
                                  epsilon=None, decay=0.0, amsgrad=False)
loss = 'mse'
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

# Shuffle the data
if shuffle == True:
    np.random.shuffle(data_list)
    
# Reduce the dataset for testing
data_list = data_list[0:num_of_train_samples]

# Load the data
X, y = datatools.load_data(path_to_dataset=path_to_dataset, 
                                               data_list=data_list, input_shape=data_shape)

# Standardize X and Scale y
if standardize_input_data == True:
    # Standardize every image in the dataset
    X = (X - X.mean(axis=(1,2,3), keepdims=True)) / X.std(axis=(1,2,3), keepdims=True)
    
    # Standardize the whole dataset
    #standardized = (X - X.mean(axis=(0,1,2), keepdims=True)) / X.std(axis=(0,1,2), keepdims=True)

y=y*data_scaling_factor

# Add the channels
X = X[:,:,:,:,np.newaxis]
y = y[:,:,:,:,np.newaxis]

# Plot some data
sample_num = 55
X_plot = X[sample_num,:,:,:,0]
y_plot = y[sample_num,:,:,:,0]

plot_slice = 15
plt.imshow(X_plot[:,:,plot_slice])

if data_scaling_factor == 1:
    print('Number of cells in y_plot: ', np.sum(y_plot))
    plt.imshow(y_plot[:,:,plot_slice])
else:
    print('Number of cells in y_plot: ', np.sum(y_plot/data_scaling_factor))
    plt.imshow(y_plot[:,:,plot_slice]/data_scaling_factor)


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

history = cnn.fit(X=X, y=y, batch_size=batch_size, epochs=epochs, callbacks=callbacks, 
                  validation_split=val_split, validation_data=None, shuffle=True)

###############################################################################
# Predict some data
###############################################################################

y_pred = cnn.predict_sample(X_plot)
y_pred = y_pred[0,:,:,:,0]

#y_pred = y_pred/32000


if data_scaling_factor == 1:
    print('Number of cells in y_pred: ', np.sum(y_pred))
    plt.imshow(y_pred[:,:,plot_slice])
else:
    print('Number of cells in y_pred: ', np.sum(y_pred/data_scaling_factor))
    plt.imshow(y_pred[:,:,plot_slice]/data_scaling_factor)
