import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import keras
import os
import time
import datetime
import tensorflow as tf
import nrrd
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

def plot_dataset_histogram(path_to_dataset, data_list):
    cell_numbers = []
    for i in range(len(data_list)):
        file = os.path.join(path_to_dataset, data_list[i])
        data, header = nrrd.read(file)
        cell_numbers.append(np.sum(data[1]))
    plt.figure()
    plt.title('Distribution of cell numbers')
    plt.hist(cell_numbers, range=(0, 25), bins=50)

#%%############################################################################
# Define the parameters
###############################################################################

# Dataset Parameters
path_to_dataset = os.path.join('..', '..', '..', 'Daten', 'dataset_size32_stride16_split')
shuffle_dataset = True # IMPORTANT that its True
plt_hist = False
clip = 5000 # Clip the patches in a range of cell numbers to this maximum number of samples, e.g. number of samples for cells in range 0,00-0,01 cells is 1000 samples
train_split = 0.07
val_split = 0.025
test_split = 0.005
data_shape = (32, 32, 32)
channels = 1

# Model Parameters
input_shape = data_shape + (channels,)
filters_exp = 4
n_filters = 16 # Number of filters for the first layer
kernel_size = (3, 3, 3)
kernel_initializer = 'glorot_uniform'
pool_size = (2, 2, 2)
hidden_layer_activation = 'relu' # 'relu', 'leaky_relu' or 'elu'
alpha = 0.3 # Only necessary if hidden_layer_activation = 'leaky_relu' or 'elu'
batchnorm = True
regularization_rate = None #0.001
dropout_rate = None # 0.3
output_layer_activation = None
padding = 'same'

# Data Generator parameters
train_shuffle = True
val_shuffle = False
standardization_mode = 'per_sample' # 'per_slice', 'per_sample' or 'per_batch'
border = None # None or border in each dimension around the inner slice which should be extracted
linear_output_scaling_factor = 819200000000

# Training parameters
#learning_rate = 0.00001
learning_rate = 0.005
epochs = 64
batch_size = 64
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


#%%############################################################################
# Define the model callbacks
###############################################################################

checkpoint = ModelCheckpoint(filepath=os.path.join(model_export_path, 'best_weights.hdf5'), 
                             monitor='val_loss', mode='min', verbose=1, save_best_only=True)

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

logdir = os.path.join('logs', timestamp)
tensor_board = TensorBoard(log_dir=logdir, histogram_freq=0, 
                              write_graph=True, write_images=True)
callbacks = [tensor_board]

#%%############################################################################
# Prepare the training data
###############################################################################

# Make a list of training files
#data_list = os.listdir(path_to_dataset)
data_list = datatools.get_balanced_dataset(path_to_dataset=path_to_dataset, clip=clip)

# Split the file list of the dataset into spheroid cultivation period
data_list_24h, data_list_48h, data_list_72h = datatools.split_cultivation_period(data_list, shuffle=shuffle_dataset)

# Split the list of training files into train validation and test data
train_24h, val_24h, test_24h = datatools.train_val_test_split(data_list_24h, train_split, 
                                                            val_split, test_split, shuffle=shuffle_dataset)
train_48h, val_48h, test_48h = datatools.train_val_test_split(data_list_48h, train_split, 
                                                            val_split, test_split, shuffle=shuffle_dataset)
train_72h, val_72h, test_72h = datatools.train_val_test_split(data_list_72h, train_split, 
                                                            val_split, test_split, shuffle=shuffle_dataset)

# Concatenate the list of training files
train_list = train_24h + train_48h + train_72h
val_list = val_24h + val_48h + val_72h
test_list = test_24h + test_48h + test_72h

# Undo the shuffeling for testing
#test_list = np.sort(test_list)
#test_list = test_list.tolist()

#%%############################################################################
# Plot the dataset distributions
###############################################################################
if plt_hist == True:
    plot_dataset_histogram(path_to_dataset=path_to_dataset, data_list=train_list)
    plot_dataset_histogram(path_to_dataset=path_to_dataset, data_list=val_list)
    plot_dataset_histogram(path_to_dataset=path_to_dataset, data_list=test_list)

#%%############################################################################
# Define the model
###############################################################################

cnn = CNN(linear_output_scaling_factor=linear_output_scaling_factor, 
          standardization_mode=standardization_mode)
#cnn.define_model(input_shape=input_shape, filters_exp=filters_exp, kernel_size=kernel_size, 
#                  pool_size=pool_size, hidden_layer_activation=hidden_layer_activation, 
#                  output_layer_activation=output_layer_activation, padding=padding, 
#                  regularization=regularization)
cnn.define_unet(input_shape=input_shape, n_filters=n_filters, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                  pool_size=pool_size, hidden_layer_activation=hidden_layer_activation, alpha=alpha, batchnorm=batchnorm,
                  regularization_rate=regularization_rate, dropout_rate=dropout_rate, output_layer_activation=None, 
                  padding='same')

#cnn.define_unet(input_shape=input_shape, n_filters=8, kernel_size=3, 
#                  batchnorm=batchnorm, hidden_layer_activation=hidden_layer_activation,
#                  output_layer_activation=None, pool_size=2, padding=padding)

summary = cnn.compile_model(loss=loss, optimizer=optimizer, metrics=metrics)

#%%############################################################################
# Fit the model
###############################################################################

train_generator = datagen.DataGenerator(val=False, path_to_dataset=path_to_dataset, 
                                        filenames_list=train_list, dim=data_shape, 
                                        channels=channels, batch_size=batch_size, shuffle=train_shuffle, 
                                        standardization_mode=standardization_mode,
                                        linear_output_scaling_factor=linear_output_scaling_factor, 
                                        border=border)
val_generator = datagen.DataGenerator(val=True, path_to_dataset=path_to_dataset, 
                                      filenames_list=val_list, dim=data_shape, 
                                      channels=channels, batch_size=batch_size, shuffle=val_shuffle, 
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
plt_slice = 16
if border == None:
    plt.imshow(X_test[plt_slice,:,:])
else:
    X_inner = impro.get_inner_slice(X_test, border)
    plt.imshow(X_inner[plt_slice,:,:])
plt.imshow(y_test[plt_slice,:,:])
plt.imshow(y_pred[plt_slice,:,:])

print('Number of cells (ground truth): ', np.sum(y_test))
print('Number of cells (predicted): ', np.sum(y_pred))
