import sys
sys.path.append("..")
import numpy as np
import nrrd
import os
import matplotlib.pyplot as plt
from tools import image_processing as impro

real_data = True

if real_data == True:
    # Get real data
    path_to_data = os.path.join('test_data', 'test_patch.nrrd')
    data, header = nrrd.read(path_to_data) # XYZ
    data = data[1,]
    data = np.transpose(data, axes=(2,1,0)) # ZYX
    shape = data.shape
    strides = (24,24,24)
    num_repeats = 5
    plt.imshow(data[9,])
else:
    # Make test data
    shape = (16,16,16)
    strides = (10,10,10)
    num_repeats = 5
    data = np.ones(shape=shape)

plot_slice = 55

# Allocate some space for the result
vol = np.zeros((num_repeats*data.shape[0], num_repeats*data.shape[1], num_repeats*data.shape[2]))

# Concat without superposition
for z in range(num_repeats):
    for y in range(num_repeats):
        for x in range(num_repeats):
            #print(z, ' ', y, ' ', x)
            z_start = z*data.shape[0]
            z_end = z_start + data.shape[0]
            y_start = y*data.shape[1]
            y_end = y_start + data.shape[1]
            x_start = x*data.shape[2]
            x_end = x_start + data.shape[2]
            vol[z_start:z_end, y_start:y_end, x_start:x_end] = data
plt.imshow(vol[plot_slice,:,:,])

# Concat with superposition
vol[:,] = 0
for z in range(num_repeats):
    for y in range(num_repeats):
        for x in range(num_repeats):
            z_start = z*strides[0]
            z_end = z_start + data.shape[0]
            y_start = y*strides[1]
            y_end = y_start + data.shape[1]
            x_start = x*strides[2]
            x_end = x_start + data.shape[2]
            vol[z_start:z_end, y_start:y_end, x_start:x_end] += data
plt.imshow(vol[plot_slice,:,:,])

# Concat with superposition and weight-matrix. If there's a homogenous volume,
# the volume after weighted superposition of overlapping patches should be 
# also homogenous.
vol[:,] = 0
for z in range(num_repeats):
    for y in range(num_repeats):
        for x in range(num_repeats):
            z_start = z*strides[0]
            z_end = z_start + data.shape[0]
            y_start = y*strides[1]
            y_end = y_start + data.shape[1]
            x_start = x*strides[2]
            x_end = x_start + data.shape[2]
            weight_matrix = impro.get_weight_matrix(shape, strides)
            data = data*weight_matrix
            vol[z_start:z_end, y_start:y_end, x_start:x_end] += data
plt.imshow(vol[plot_slice,:,:,])

# Concat with superposition and weight-matrix
