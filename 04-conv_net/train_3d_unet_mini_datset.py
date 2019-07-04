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

###############################################################################
# Define the parameters
###############################################################################

# Dataset Parameters
data_shape = (32, 32, 32)
channels = 1

# Model Parameters
input_shape = data_shape + (channels,)
hidden_layer_activation = 'relu'
#hidden_layer_activation = keras.layers.LeakyReLU(alpha=0.2)
output_layer_activation = None
padding = 'same'

# Data Generator parameters
train_shuffle = True
val_shuffle = False
standardization_mode = 'per_sample' # 'per_slice', 'per_sample' or 'per_batch'
border = None # None or border in each dimension around the inner slice which should be extracted
linear_output_scaling_factor = 409600000000

# Training parameters
#learning_rate = 0.00001
learning_rate = 0.001
epochs = 32
batch_size = 32
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
tensor_board = TensorBoard(log_dir=logdir, histogram_freq=0, 
                              write_graph=True, write_images=True)
callbacks = [tensor_board]

###############################################################################
# Prepare the training data
###############################################################################

path_to_trainset = os.path.join('..', '..', '..', 'Daten', 'dataset_mini', 'train')
path_to_valset = os.path.join('..', '..', '..', 'Daten', 'dataset_mini', 'val')
path_to_testset = os.path.join('..', '..', '..', 'Daten', 'dataset_mini', 'test')

train_list = os.listdir(path_to_trainset)
val_list = os.listdir(path_to_valset)
test_list = os.listdir(path_to_testset)

###############################################################################
# Define the model
###############################################################################

cnn = CNN(linear_output_scaling_factor=linear_output_scaling_factor, 
          standardization_mode=standardization_mode)
cnn.define_model(input_shape=input_shape, filters_exp=5, kernel_size=(3, 3, 3), 
                  pool_size=(2, 2, 2), hidden_layer_activation=hidden_layer_activation, 
                  output_layer_activation=output_layer_activation, padding=padding)

cnn.compile_model(loss=loss, optimizer=optimizer, metrics=metrics)

###############################################################################
# Fit the model
###############################################################################

train_generator = datagen.DataGenerator(path_to_dataset=path_to_trainset, 
                                        filenames_list=train_list, dim=data_shape, 
                                        channels=channels, batch_size=batch_size, shuffle=train_shuffle, 
                                        standardization_mode=standardization_mode,
                                        linear_output_scaling_factor=linear_output_scaling_factor, 
                                        border=border)
val_generator = datagen.DataGenerator(path_to_dataset=path_to_valset, 
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
X_test_data, y_test_data = datatools.load_data(path_to_dataset=path_to_testset, 
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