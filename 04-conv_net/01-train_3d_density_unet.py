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
from tools import datagen
from tools import image_processing as impro
from tools import datatools

def plot_dataset_histogram(path_to_dataset, data_list, hist_export_file=None):
    cell_numbers = []
    for i in range(len(data_list)):
        file = os.path.join(path_to_dataset, data_list[i])
        data, header = nrrd.read(file)
        cell_number = np.sum(data[1])
        cell_numbers.append(cell_number)
        if cell_number > 25.0:
            print('Warning: Cell number in patch > 25 cells in ', file)
    plt.figure()
    plt.title('Distribution of cell numbers')
    plt.hist(cell_numbers, range=(0, 15), bins=50)
    if hist_export_file != None:
        plt.savefig(hist_export_file, dpi=300)
        

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
    
test_spheroids = {'untreated': [ '24h_C2-untreated_2.2',
                                '48h_C2-untreated_4.1',
                                '72h_C2-untreated_4']}
    
# Specify the number of spheroids per dataset
num_train_spheroids = 4
num_val_spheroids = 1
num_test_spheroids = 1


# Dataset Parameters
path_to_dataset = os.path.join('..', '..', '..', 'Datensaetze', 'dataset_density-maps_fiji_and_mathematica')
plt_hist = True
data_shape = (32, 32, 32)
channels = 1
train_percentage = 1
val_percentage = 1
test_percentage = 1

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
output_layer_activation = None
upsampling_method = 'Conv3DTranspose' #'Conv3DTranspose' or 'UpSampling3D'
padding = 'same'

# Data Generator parameters
train_shuffle = True
val_shuffle = False
standardization_mode = 'per_sample' # 'per_slice', 'per_sample' or 'per_batch'
border = None # None or border in each dimension around the inner slice which should be extracted
#linear_output_scaling_factor = 1e11#1e12#409600000000
linear_output_scaling_factor = 1e5#1e12#409600000000

# Training parameters
#learning_rate = 0.00001
#learning_rate = 0.005
learning_rate = 0.005
epochs = 128
batch_size = 256
optimizer = keras.optimizers.adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, 
                                  epsilon=None, decay=0.0, amsgrad=False)
#optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=0., decay=0., nesterov=False)
evaluate = True

#optimizer = keras.optimizers.SGD(lr=learning_rate)
loss = ['mse']
metrics = ['mae']
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
# Example if you want to choose the spheroids used for train val and testing randomized
train_files, val_files, test_files, dataset_table = datatools.get_datasets(path_to_dataset=path_to_dataset, spheroid_names=spheroid_names_dataset_all, 
                                                                 num_train_spheroids=num_train_spheroids, num_val_spheroids=num_val_spheroids, 
                                                                 num_test_spheroids=num_test_spheroids, train_spheroids=None, 
                                                                 val_spheroids=None, test_spheroids=None)

# Example if you want to choose the spheroids used for train val and testing manually
#train_files, val_files, test_files, dataset_table = datatools.get_datasets(path_to_dataset=path_to_dataset, spheroid_names=spheroid_names_dataset_all, 
#                                                                 num_train_spheroids=num_train_spheroids, num_val_spheroids=num_val_spheroids, 
#                                                                 num_test_spheroids=num_test_spheroids, train_spheroids=train_spheroids, 
#                                                                 val_spheroids=val_spheroids, test_spheroids=test_spheroids)

# Save the spheroid names used for training, validation and testing in a table
table_export_path = os.path.join(model_export_path, 'train_val_test-spheroids.txt')

with open(table_export_path,'w+') as file:
    for item in dataset_table:
        line = "%s \t %s \t %s\n" %(item[0], item[1], item[2])
        file.write(line)

np.random.shuffle(train_files)
train_files = train_files[0:int(train_percentage*len(train_files))]

np.random.shuffle(val_files)
val_files = val_files[0:int(val_percentage*len(val_files))]

np.random.shuffle(test_files)
test_files = test_files[0:int(test_percentage*len(test_files))]

print('Starting...')
#%%############################################################################
# Plot the dataset distributions
###############################################################################
if plt_hist == True:
    plot_dataset_histogram(path_to_dataset=path_to_dataset, data_list=train_files, hist_export_file=os.path.join(model_export_path, 'train_hist.png'))
    plot_dataset_histogram(path_to_dataset=path_to_dataset, data_list=val_files, hist_export_file=os.path.join(model_export_path, 'val_hist.png'))
    plot_dataset_histogram(path_to_dataset=path_to_dataset, data_list=test_files, hist_export_file=os.path.join(model_export_path, 'test_hist.png'))

###############################################################################
# Define the model
###############################################################################
cnn = CNN(linear_output_scaling_factor=linear_output_scaling_factor, 
          standardization_mode=standardization_mode)

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
train_generator = datagen.DataGenerator(val=False, path_to_dataset=path_to_dataset, 
                                        filenames_list=train_files, dim=data_shape, 
                                        channels=channels, batch_size=batch_size, shuffle=train_shuffle, 
                                        standardization_mode=standardization_mode,
                                        linear_output_scaling_factor=linear_output_scaling_factor, 
                                        border=border)
val_generator = datagen.DataGenerator(val=True, path_to_dataset=path_to_dataset, 
                                      filenames_list=val_files, dim=data_shape, 
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
                                               data_list=test_files, input_shape=data_shape,
                                               standardization_mode=None,
                                               border=border)
if evaluate == True:
    # Evaluate the model on the test-data
    test_loss = cnn.evaluate_model(X_test=np.expand_dims(X_test_data, axis=4), 
                       y_test=np.expand_dims(y_test_data, axis=4), batch_size=batch_size)
    
    # Export the value of the test-loss
    test_loss_export_path = os.path.join(model_export_path, 'test_loss.txt')
    with open(test_loss_export_path,'w') as file:
        for l in range(len(test_loss)):
            file.write(str(test_loss[l])+'\n')
    


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
