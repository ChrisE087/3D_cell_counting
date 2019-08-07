import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import os
import nrrd
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tools.cnn import CNN
from tools import datagen
from tools import image_processing as impro
from tools import datatools

#%%############################################################################
# Specify the parameters
###############################################################################

# Specify the patch sizes and strides in each direction (ZYX)
patch_sizes = (32, 32, 32)
strides = (32, 32, 32)

# Specify the border around a patch in each dimension (ZYX), which is removed
cut_border = None #(8,8,8)

# Specify the padding which is used for the prediction of the patches
padding = 'VALID'

# Specify which model is used
model_import_path = os.path.join(os.getcwd(), 'model_export', 'BEST', '2019-08-02_14-08-13_100000.0')

# Specify the standardization mode
standardization_mode = 'per_sample'

# Specify the linear output scaling factor !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
linear_output_scaling_factor = 1e5#1e11#409600000000

# Specify if the results are saved
save_results = True

#%%############################################################################
# Read the data
###############################################################################
#category = '24h'
#spheroid_name = 'C1-untreated_1.2.nrrd'
#path_to_nuclei = os.path.join('..', '..', '..', 'Daten', category, 'untreated', spheroid_name)

category = 'Fibroblasten'
spheroid_name = '5_draq5.nrrd'
path_to_spheroid = os.path.join('..', '..', '..', 'Daten2', category, spheroid_name)

#%%############################################################################
# Initialize the CNN
###############################################################################
cnn = CNN(linear_output_scaling_factor=linear_output_scaling_factor, 
          standardization_mode=standardization_mode)
cnn.load_model_json(model_import_path, 'model_json', 'best_weights')

#%%############################################################################
# Predict the density-map
###############################################################################
spheroid_new, density_map, num_of_cells = cnn.predict_spheroid(path_to_spheroid=path_to_spheroid, patch_sizes=patch_sizes, 
                                                               strides=strides, border=cut_border, padding=padding)
plt.figure()
plt.imshow(spheroid_new[int(spheroid_new.shape[0]/2),])
plt.figure()
plt.imshow(density_map[int(density_map.shape[0]/2),])
print('Number of cells = ', num_of_cells)

#%%############################################################################
# Save the results
###############################################################################
if save_results == True:
    nrrd.write(category+'-'+spheroid_name+'.nrrd', spheroid_new)
    nrrd.write(category+'-'+spheroid_name+'-density_map.nrrd', density_map)
