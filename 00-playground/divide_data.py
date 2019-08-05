import nrrd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import sys
sys.path.append("..")



def divide_data(data, dim_order='XYZ'):
    if dim_order == 'ZYX':
        data = np.copy(data)
        data = np.transpose(data, axes=(2,1,0))#XYZ
    z_start = 0
    z_middle = int(np.floor(data.shape[2]/2))
    z_end = data.shape[2]
    north = data[0:, 0:, z_start:z_middle]
    south = data[0:, 0:, z_middle+1:z_end]
    return north, south
    
    
data, header = nrrd.read('test_data/test_input_data.nrrd')
plt.imshow(data[:,:,50])

north, south = divide_data(data)
plt.imshow(north[:,:,0])
plt.imshow(north[:,:,north.shape[2]-1])
plt.imshow(south[:,:,0])
plt.imshow(south[:,:,south.shape[2]-1])

