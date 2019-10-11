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
strides = (28, 28, 28)
#strides = (16, 16, 16)

# Specify the border around a patch in each dimension (ZYX), which is removed
cut_border = (2, 2, 2) #None #(8,8,8)
#cut_border = (8, 8, 8)

# Specify the padding which is used for the prediction of the patches
padding = 'SAME'

# Specify which model is used
model_import_path = os.path.join('..', '..', '..', 'Ergebnisse', 'CNNs', 'Universalnetze', 'density_maps', '2019-08-12_14-42-57_100000.0_3_train_samples_fiji_and_mathematica_densitymaps')

# Specify the standardization mode
standardization_mode = 'per_sample'

# Specify the linear output scaling factor !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
linear_output_scaling_factor = 1e5#1e11#409600000000

# Specify if the results are saved
save_results = False

#%%############################################################################
# Read the data
###############################################################################
#category = '24h'
#category = '48h'
#category = '72h'
#spheroid_name = 'C2-untreated_1.nrrd'
#path_to_spheroid = os.path.join('..', '..', '..', 'Datensaetze', 'Aufnahmen_und_Segmentierungen', 'Datensatz1', category, 'untreated', spheroid_name)

#category = 'Fibroblasten'
#category = 'Hacat'
#category = 'HT29'
#category = 'HTC8'
#category = 'NPC1'
#spheroid_name = 'C3-7.nrrd'
#path_to_spheroid = os.path.join('..', '..', '..', 'Datensaetze', 'Aufnahmen_und_Segmentierungen', 'Datensatz2', category, spheroid_name)

category = 'none'
spheroid_name = 'X_scaled.nrrd'
path_to_spheroid = os.path.join('..', '..', '..', 'Datensaetze', 'OpenSegSPIM_Beispieldaten', 'Neurosphere', spheroid_name)

#path_to_spheroid = os.path.join('Skalierung', 'NPC1', 'C3-2-1_1-3_upper.nrrd')

#%%############################################################################
# Initialize the CNN
###############################################################################
cnn = CNN(linear_output_scaling_factor=linear_output_scaling_factor, 
          standardization_mode=standardization_mode)
cnn.load_model_json(model_import_path, 'model_json', 'best_weights')

#%%############################################################################
# Predict the density-map
###############################################################################
spheroid_new, density_map, num_of_cells = cnn.predict_density_map(path_to_spheroid=path_to_spheroid, patch_sizes=patch_sizes, 
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
    export_name = spheroid_name[0:-5]
    nrrd.write(category+'-'+export_name+'-nuclei.nrrd', spheroid_new)
    nrrd.write(category+'-'+export_name+'-density_map.nrrd', density_map)
