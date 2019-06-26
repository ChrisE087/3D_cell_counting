import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import keras
import os
import time
import datetime
import nrrd
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tools.cnn import CNN
from tools import datagen
from tools import image_processing as impro
from tools import datatools

# Read the data
path_to_nuclei = os.path.join('..', '..', '..', 'Daten', '24h', 'untreated', 'C1-untreated_1.1.nrrd')
data, header = nrrd.read(path_to_nuclei)
data = data.astype(np.float32)

plt.imshow(data[:,:,55])

# Load the CNN
linear_output_scaling_factor = 2048000
cnn = CNN(linear_output_scaling_factor)
import_path = os.path.join(os.getcwd(), 'model_export', '2019-06-25_10-15-30')
cnn.load_model_json(import_path, 'model_json', 'model_weights')

# Generate image patches
size_z = patch_slices = 32
size_y = patch_rows = 32
size_x = patch_cols = 32
stride_z = stride_slices = 32
stride_y = stride_rows = 32
stride_x = stride_cols = 32
patches = impro.gen_patches(data=data, patch_slices=size_z, patch_rows=size_y, 
                            patch_cols=size_x, stride_slices=stride_z, stride_rows=stride_y, 
                            stride_cols=stride_x, input_dim_order='XYZ', padding='VALID')

#p = patches[2,15,6,15,:,:]
#plt.imshow(p)
predictions = np.zeros_like(patches, dtype=np.float32)

# Predict the density-patches
for zslice in range(patches.shape[0]):
    for row in range(patches.shape[1]):
        for col in range(patches.shape[2]):
            X = patches[zslice, row, col, :]
            prediction = cnn.predict_sample(X)
            predictions[zslice, row, col, :] = prediction

## Restore the volumes from the patches
nuclei = impro.restore_volume(patches=patches, output_dim_order='XYZ')         
density_map = impro.restore_volume(patches=predictions, output_dim_order='XYZ')

#
## Plot patch
#pz = 0
#py = 0
#px = 0  
#s = 12
#
#X = patches[pz, py, px, s, :, :]
#y = predictions[pz, py, px, s, :, :]
#plt.imshow(X)
#plt.imshow(y)
#
## Print the sum of the density-patch
#print (np.sum(y))
#
## Print the sum of the whole density-map
print(np.sum(density_map))
plt.imshow(density_map[:,:,150])
            

# Save the results
nrrd.write('nuclei.nrrd', nuclei)
nrrd.write('density_map.nrrd', density_map)