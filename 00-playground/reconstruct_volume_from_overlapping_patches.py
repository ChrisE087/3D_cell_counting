import sys
sys.path.append("..")
import numpy as np
import nrrd
import os
import matplotlib.pyplot as plt
from tools import image_processing as impro

# Load the data
path_to_data = os.path.join('test_data', 'kernel_groß.nrrd')
data, header = nrrd.read(path_to_data)
#data = np.transpose(data, axes=(1,0,2))
plt.imshow(data[:,:,20])

# Specify patch size
patch_slices = 4
patch_rows = 16
patch_cols = 16

# Specify strides
stride_slices = 2
stride_rows = 4
stride_cols = 4
patches = impro.gen_patches(data=data, patch_slices=patch_slices, 
                            patch_rows=patch_rows, patch_cols=patch_cols, 
                            stride_slices=stride_slices, 
                            stride_rows=stride_rows, stride_cols=stride_cols, 
                            input_dim_order='XYZ', padding='SAME')

plt.imshow(patches[2,1,1,3,:,:]) # Patch index=ZYX, Patch-dimension order = ZYX


#reconstructed = np.zeros(shape=(patches.shape[0]*patches.shape[3]/stride_dim0, 
#                                patches.shape[1]*patches.shape[4]/stride_dim1, 
#                                patches.shape[2]*patches.shape[5]/stride_dim2))

# Calculate the size of the reconstructed volume (not so clear how TensorFlow
# calculates the Padding) -> savety distance
safety_distance = 4
if patch_slices / stride_slices == 1:
    out_dim0 = data.shape[0]
else:
    out_dim0 = np.int(np.ceil(patch_slices/2)+patches.shape[0]*stride_slices) + safety_distance

if patch_rows / stride_rows == 1:
    out_dim1 = data.shape[1]
else:
    out_dim1 = np.int(np.ceil(patch_rows/2)+patches.shape[1]*stride_rows) + safety_distance

if patch_cols / stride_cols == 1:
    out_dim2 = data.shape[2]
else:
    out_dim2 = np.int(np.ceil(patch_cols/2)+patches.shape[2]*stride_cols) + safety_distance

# Allocate Volume for reconstruction
reconstructed = np.zeros(shape=(out_dim0, out_dim1, out_dim2))
#reconstructed = np.zeros(shape=(out_dim0, out_dim1, out_dim2))

for dim0 in range(patches.shape[0]):
    for dim1 in range(patches.shape[1]):
        for dim2 in range(patches.shape[2]):
            # Extract patch
            patch = patches[dim0, dim1, dim2]
            # Calculate the position to place this patch in the reconstructed volume
            start_dim0 = dim0*stride_slices
            end_dim0 = start_dim0 + patch_slices
            start_dim1 = dim1*stride_rows
            end_dim1 = start_dim1 + patch_rows
            start_dim2 = dim2*stride_cols
            end_dim2 = start_dim2 + patch_cols
            rec = reconstructed[start_dim0:end_dim0, start_dim1:end_dim1, start_dim2:end_dim2]
#            print('patch: ', patch.shape)
#            print('reconstructed: ', rec.shape)
#            
#            print('Placing patch to\nDim0: ', start_dim0, ' to ', end_dim0, '\n',
#                  'Dim1: ', start_dim1, ' to ', end_dim1, '\n',
#                  'Dim2: ', start_dim2, ' to ', end_dim2, '\n')
            reconstructed[start_dim0:end_dim0, start_dim1:end_dim1, start_dim2:end_dim2] += patch 
            reconstructed[start_dim0:end_dim0, start_dim1:end_dim1, start_dim2:end_dim2] /2
            
            #plt.figure()
            #plt.imshow(patch[:,:,1])


reconstructed = impro.restore_volume_from_overlapped_patches(patches, (127,127,127), (stride_slices, stride_rows, stride_cols))
            
nrrd.write('test.nrrd', np.transpose(reconstructed, axes=(2,1,0)))
