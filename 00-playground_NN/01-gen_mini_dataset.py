import nrrd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import sys
sys.path.append("..")
from tools import image_processing as impro

def save_dataset(p_X, p_Y, export_path):
    num = 0
    for dim0 in range(p_X.shape[0]):
        for dim1 in range(p_X.shape[1]):
            for dim2 in range(p_X.shape[2]):
                export = np.zeros((2, p_X.shape[3], p_X.shape[4], p_X.shape[5]))
                X = p_X[dim0, dim1, dim2,:,:,:]
                Y = p_Y[dim0, dim1, dim2,:,:,:]
                tresh = 0.0
                #if np.sum(Y) > tresh:
                export[0,] = X
                export[1,] = Y
                patch_name = "%06d.nrrd" % (num)
                num += 1
                filename = os.path.join(export_path, patch_name)
                nrrd.write(filename, export)
        
#%%############################################################################
# Extract the dataset from the spheroid
###############################################################################
path_X = os.path.join('test_data', '48h-X-C2-untreated_3.nrrd')
path_Y = os.path.join('test_data', '48h-Y_Fiji-C2-untreated_3.nrrd')
X, X_header = nrrd.read(path_X) # XYZ
Y, y_header = nrrd.read(path_Y) # XYZ

extract = 'upper_left' # extract the 'upper_left' or 'center' of the spheroid

# Print the min and max after scaling
Y_s = Y*819200000000
print(np.min(Y_s))
print(np.max(Y_s))
print(X.shape)

# Specify the extracted size of the mini dataset
size_X = 200
size_Y = 200
size_Z = 200

if extract == 'center':
    # Calculate the center of the volume
    c_X = int(X.shape[0]/2)
    c_Y = int(X.shape[1]/2)
    c_Z = int(X.shape[2]/2)
elif extract == 'upper_left':
    # Set the center of the volume to the upper left
    c_X = 110
    c_Y = 110
    c_Z = 250
else:
    print('Wrong argument for "extract"')

# Slice a volume out of the center
X_s = X[c_X-int(size_X/2):c_X+int(size_X/2), 
        c_Y-int(size_Y/2):c_Y+int(size_Y/2), 
        c_Z-int(size_Z/2):c_Z+int(size_Z/2)]
Y_s = Y[c_X-int(size_X/2):c_X+int(size_X/2), 
        c_Y-int(size_Y/2):c_Y+int(size_Y/2), 
        c_Z-int(size_Z/2):c_Z+int(size_Z/2)]

# Plot a slice of the dataset
plt.imshow(X_s[:,:,20])
plt.imshow(Y_s[:,:,20])

#%%############################################################################
# Generate the patches
###############################################################################
patch_slices = patch_rows = patch_cols = 32
stride_slices = stride_rows = stride_cols = 16

# WORKAROUND because tensorflow moves the X and y volumes relative to each 
# other if X are int values and Y are float values -> Normalize and Scale the 
# target data with datatype int
Y_s, max_val, min_val = impro.normalize_data(Y_s)
Y_s = impro.scale_data(Y_s, 65535)

# Extract the image patches
session = tf.Session()
p_X = impro.gen_patches(session=session, data=X_s, patch_slices=patch_slices, 
                   patch_rows=patch_rows, patch_cols=patch_cols, 
                   stride_slices=stride_slices, stride_rows=stride_rows, 
                   stride_cols=stride_cols, input_dim_order='XYZ', 
                   padding='SAME')
p_Y = impro.gen_patches(session=session, data=Y_s, patch_slices=patch_slices, 
                   patch_rows=patch_rows, patch_cols=patch_cols, 
                   stride_slices=stride_slices, stride_rows=stride_rows, 
                   stride_cols=stride_cols, input_dim_order='XYZ', 
                   padding='SAME')

# WORKAROUND undo the normalization and scaling
p_Y = impro.unscale_data(p_Y, 65535)
p_Y = impro.unnormalize_data(p_Y, max_val, min_val)

# Plot a slice of an example patch
plt.imshow(p_X[4,5,6,8,:,:])
plt.imshow(p_Y[4,5,6,8,:,:])

export_path = os.path.join(os.getcwd(), 'mini_dataset', 'Fiji_upper_left_size32_stride16')
save_dataset(p_X, p_Y, export_path)

