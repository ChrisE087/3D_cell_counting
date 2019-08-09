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

def plot_dataset_histogram(path_to_dataset, data_list):
    cell_numbers = []
    for i in range(len(data_list)):
        file = os.path.join(path_to_dataset, data_list[i])
        data, header = nrrd.read(file)
        cell_numbers.append(np.sum(data[1]))
    plt.figure()
    plt.title('Distribution of cell numbers')
    plt.hist(cell_numbers, range=(0, 25), bins=50)

###############################################################################
# Define the parameters
###############################################################################

# Dataset Parameters
path_to_trainset = os.path.join('..', '..', '..', 'Datensaetze', 'dataset1_segmentation_fiji', 'train')
path_to_valset = os.path.join('..', '..', '..', 'Datensaetze', 'dataset1_segmentation_fiji', 'val')
path_to_testset = os.path.join('..', '..', '..', 'Datensaetze', 'dataset1_segmentation_fiji', 'test')
plt_hist = False
data_shape = (32, 32, 32)
channels = 1
train_percentage = 1
val_percentage = 1

# Model Parameters
input_shape = data_shape + (channels,)
#filters_exp = 4 # 2^filters_exp filters for the first layer
n_filters = 4 # Number of filters for the first layer
kernel_size = (3, 3, 3)
kernel_initializer = 'glorot_uniform'#'he_normal'
pool_size = (2, 2, 2)
hidden_layer_activation = 'relu'
#hidden_layer_activation = keras.layers.LeakyReLU(alpha=0.2)
alpha = 0.3 # Only necessary if hidden_layer_activation = 'leaky_relu' or 'elu'
batchnorm_encoder = False
batchnorm_decoder = False
regularization_rate = None#0.001#None #0.001
dropout_rate = None # 0.3
output_layer_activation = 'sigmoid'
upsampling_method = 'Conv3DTranspose' #'Conv3DTranspose' or 'UpSampling3D'
padding = 'same'

# Data Generator parameters
train_shuffle = True
val_shuffle = False
standardization_mode = 'per_sample' # 'per_slice', 'per_sample' or 'per_batch'
border = None # None or border in each dimension around the inner slice which should be extracted
#linear_output_scaling_factor = 1e11#1e12#409600000000
linear_output_scaling_factor = 1

# Training parameters
#learning_rate = 0.00001
#learning_rate = 0.005
learning_rate = 0.005
epochs = 128
batch_size = 256
optimizer = keras.optimizers.adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, 
                                  epsilon=None, decay=0.0, amsgrad=False)
#optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=0., decay=0., nesterov=False)
evaluate = False

#optimizer = keras.optimizers.SGD(lr=learning_rate)
loss = [keras.losses.binary_crossentropy]
metrics = [dice_coef_loss]
load_model = False

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
timestamp = timestamp+'_'+str(linear_output_scaling_factor)
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
callbacks = [tensor_board_cb, checkpoint]

###############################################################################
# Prepare the training data
###############################################################################

train_list = os.listdir(path_to_trainset)
np.random.shuffle(train_list)
train_list = train_list[0:int(train_percentage*len(train_list))]

val_list = os.listdir(path_to_valset)
np.random.shuffle(val_list)
val_list = val_list[0:int(val_percentage*len(val_list))]

test_list = os.listdir(path_to_testset)

#%%############################################################################
# Plot the dataset distributions
###############################################################################
if plt_hist == True:
    plot_dataset_histogram(path_to_dataset=path_to_trainset, data_list=train_list)
    plot_dataset_histogram(path_to_dataset=path_to_valset, data_list=val_list)
    plot_dataset_histogram(path_to_dataset=path_to_testset, data_list=test_list)

###############################################################################
# Define the model
###############################################################################

cnn = CNN(linear_output_scaling_factor=linear_output_scaling_factor, 
          standardization_mode=standardization_mode)
#cnn.define_model(input_shape=input_shape, filters_exp=filters_exp, kernel_size=kernel_size, 
#                  pool_size=pool_size, hidden_layer_activation=hidden_layer_activation, 
#                  output_layer_activation=output_layer_activation, padding=padding)
cnn.define_unet(input_shape=input_shape, n_filters=n_filters, kernel_size=kernel_size, 
                  pool_size=pool_size, kernel_initializer=kernel_initializer, 
                  hidden_layer_activation=hidden_layer_activation, alpha=alpha, 
                  batchnorm_encoder=batchnorm_encoder, batchnorm_decoder=batchnorm_decoder,
                  regularization_rate=regularization_rate, dropout_rate=dropout_rate, 
                  output_layer_activation=output_layer_activation, 
                  upsampling_method=upsampling_method, padding=padding)

summary = cnn.compile_model(loss=loss, optimizer=optimizer, metrics=metrics)

###############################################################################
# Fit the model
###############################################################################

train_generator = datagen.DataGenerator(val=False, path_to_dataset=path_to_trainset, 
                                        filenames_list=train_list, dim=data_shape, 
                                        channels=channels, batch_size=batch_size, shuffle=train_shuffle, 
                                        standardization_mode=standardization_mode,
                                        linear_output_scaling_factor=linear_output_scaling_factor, 
                                        border=border)
val_generator = datagen.DataGenerator(val=True, path_to_dataset=path_to_valset, 
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
X_test_data, y_test_data = datatools.load_data(path_to_dataset=path_to_valset, 
                                               data_list=val_list, input_shape=data_shape,
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
