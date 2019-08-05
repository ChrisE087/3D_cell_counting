import nrrd
import numpy as np
import os

data, header = nrrd.read(os.path.join('test_data', 'test_input_data.nrrd')) #XYZ
data = np.transpose(data, axes=(2,1,0)) #ZYX

padding = 'same'
patch_size = np.array((32,32,32)) #ZYX
strides = np.array((32,32,32)) #ZYX






if padding == 'same':
    # Calculate the padding
    out_z = np.ceil(float(data.shape[0]) / float(strides[0]))
    out_y = np.ceil(float(data.shape[1]) / float(strides[1]))
    out_x  = np.ceil(float(data.shape[2]) / float(strides[2]))
    
    pad_along_z = max((out_z - 1) * strides[0] + patch_size[0] - data.shape[0], 0).astype(np.int)
    pad_along_y = max((out_y - 1) * strides[1] + patch_size[1] - data.shape[1], 0).astype(np.int)
    pad_along_x = max((out_x - 1) * strides[2] + patch_size[2] - data.shape[2], 0).astype(np.int)
    
    pad_front = int(pad_along_z // 2)
    pad_back = int(pad_along_z - pad_front)
    pad_top = int(pad_along_y // 2)
    pad_bottom = int(pad_along_y - pad_top)
    pad_left = int(pad_along_x // 2)
    pad_right = int(pad_along_x - pad_left)
    
    # Pad the data
    data_p = np.zeros(shape=(data.shape[0]+pad_along_z, data.shape[1]+pad_along_y, data.shape[2]+pad_along_x))
    data_p[pad_front:pad_front+data.shape[0], pad_top:pad_top+data.shape[1], pad_left:pad_left+data.shape[2]] = data

data = data_p

patches_z = int(data.shape[0] / strides[0])
patches_y = int(data.shape[1] / strides[1])
patches_x = int(data.shape[2] / strides[2])
patches = np.zeros(shape=(patches_z, patches_y, patches_x, patch_size[0], patch_size[1], patch_size[2]))

idx_z = idx_y = idx_x = 0
for z in range(patches_z):
    for y in range(patches_y):
        for x in range(patches_x):
            z_start = idx_z*strides[0]
            z_end = z_start + patch_size[0]
            y_start = idx_y*strides[1]
            y_end = y_start + patch_size[1]
            x_start = idx_x*strides[2]
            x_end = x_start + patch_size[2]
            patches[idx_z, idx_y, idx_x,:,:,:] = data[z_start:z_end, y_start:y_end, x_start:x_end]
            
            idx_x = idx_x + 1
        idx_y = idx_y + 1
    idx_z = idx_z + 1