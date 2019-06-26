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

def mse_inner_slice(y_true, y_pred):
    print('Shape y_true: ', y_true.shape)
    print('Shape y_true: ', y_pred.shape)
#    y_true = impro.get_inner_slice(data=y_true[0,:,:,:,0], border=(16,16,16))
#    y_pred = impro.get_inner_slice(data=y_pred[0,:,:,:,0], border=(16,16,16))
#    y_true = y_true[np.newaxis,:,:,:,np.newaxis]
#    y_pred = y_pred[np.newaxis,:,:,:,np.newaxis]
    return keras.losses.mean_squared_error(y_true, y_pred)

###############################################################################
# Define the parameters
###############################################################################

# Dataset Parameters
path_to_dataset = os.path.join('..', '..', '..', 'Daten', 'dataset_mini')
train_split = 0.8
val_split = 0.1
test_split = 0.1
data_shape = (64, 64, 64)
channels = 1

# Model Parameters
input_shape = data_shape + (channels,)
hidden_layer_activation = 'relu'
#hidden_layer_activation = keras.layers.LeakyReLU(alpha=0.2)
output_layer_activation = None
padding = 'same'

# Data Generator parameters
shuffle = True
normalize_input_data = False
standardize_input_data = True
standardization_mode = 'volume_wise'
border = (16,16,16)
linear_output_scaling_factor = 4096000

# Training parameters
#learning_rate = 0.00001
learning_rate = 0.0001
epochs = 128
batch_size = 8
optimizer = keras.optimizers.adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, 
                                  epsilon=None, decay=0.0, amsgrad=False)
evaluate = False

#optimizer = keras.optimizers.SGD(lr=learning_rate)
loss = ['mse']
metrics = ['mae']
load_model = False

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
model_export_path = os.path.join(os.getcwd(), 'model_export', timestamp)
if not os.path.exists(model_export_path):
    os.makedirs(model_export_path)


###############################################################################
# Define the model callbacks
###############################################################################

checkpoint = ModelCheckpoint(filepath=os.path.join(model_export_path, 'best_weights.hdf5'), 
                             monitor='val_loss', mode='min', verbose=1, save_best_only=True)

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

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

cnn = CNN(linear_output_scaling_factor=linear_output_scaling_factor, 
          standardize_input_data=standardize_input_data,
          standardization_mode=standardization_mode)
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
                                        standardization_mode=standardization_mode,
                                        linear_output_scaling_factor=linear_output_scaling_factor, 
                                        border=border)
val_generator = datagen.DataGenerator(path_to_dataset=path_to_dataset, 
                                      filenames_list=val_list, dim=data_shape, 
                                      channels=channels, batch_size=batch_size, shuffle=shuffle, 
                                      normalize_input_data=normalize_input_data, 
                                      standardize_input_data=standardize_input_data,
                                      standardization_mode=standardization_mode,
                                      linear_output_scaling_factor=linear_output_scaling_factor, 
                                      border=border)

history = cnn.fit_generator(epochs=epochs, train_generator=train_generator, val_generator=val_generator, 
                       callbacks=callbacks)

###############################################################################
# Evaluate the model
###############################################################################
# Load unstandardized test data
X_test_data, y_test_data = datatools.load_data(path_to_dataset=path_to_dataset, 
                                               data_list=test_list, input_shape=data_shape,
                                               standardize_input_data=standardize_input_data,
                                               standardization_mode=None,
                                               border=border)
if evaluate == True:
    test_loss = cnn.evaluate_model(X_test=np.expand_dims(X_test_data, axis=4), 
                       y_test=np.expand_dims(y_test_data, axis=4), batch_size=batch_size)
    print(test_loss)


###############################################################################
# Save the model
###############################################################################

cnn.save_model_json(model_export_path, 'model_json')
cnn.save_model_weights(model_export_path, 'model_weights')
cnn.save_model_single_file(model_export_path, 'model_single')

###############################################################################
# Load the model
###############################################################################

if load_model == True:
    import_path = os.path.join(os.getcwd(), 'model_export', '2019-06-12_21-46-36')
    #cnn.load_model_single_file(import_path, 'model_single')
    cnn.load_model_json(import_path, 'model_json', 'model_weights')

###############################################################################
# Predict some data
###############################################################################
rand_int = np.random.randint(low=0, high=np.size(X_test_data, axis=0))
X_test = X_test_data[rand_int,]
y_test = y_test_data[rand_int,]

y_pred = cnn.predict_sample(X_test)

if border == None:
    plt.imshow(X_test[:,:,16])
else:
    X_inner = impro.get_inner_slice(X_test, border)
    plt.imshow(X_inner[:,:,16])
plt.imshow(y_test[:,:,16])
plt.imshow(y_pred[:,:,16])

print('Number of cells (ground truth): ', np.sum(y_test))
print('Number of cells (predicted): ', np.sum(y_pred))
