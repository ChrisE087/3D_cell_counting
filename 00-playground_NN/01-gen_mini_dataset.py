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
                patch_name = "%04d.nrrd" % (num)
                num += 1
                filename = os.path.join(export_path, patch_name)
                nrrd.write(filename, export)
        

path_X = os.path.join('test_data', '48h-X-C2-untreated_3.nrrd')
path_Y = os.path.join('test_data', '48h-Y-C2-untreated_3.nrrd')
X, X_header = nrrd.read(path_X) # XYZ
Y, y_header = nrrd.read(path_Y) # XYZ

Y_s = Y*409600000000
print(np.min(Y_s))
print(np.max(Y_s))
print(X.shape)

# Specify the extracted size of the mini dataset
size_X = 200
size_Y = 200
size_Z = 200

# Calculate the center of the volume
#c_X = int(X.shape[0]/2)
#c_Y = int(X.shape[1]/2)
#c_Z = int(X.shape[2]/2)

c_X = 110
c_Y = 110
c_Z = 250

# Slice a volume out of the center
X_s = X[c_X-int(size_X/2):c_X+int(size_X/2), 
        c_Y-int(size_Y/2):c_Y+int(size_Y/2), 
        c_Z-int(size_Z/2):c_Z+int(size_Z/2)]
Y_s = Y[c_X-int(size_X/2):c_X+int(size_X/2), 
        c_Y-int(size_Y/2):c_Y+int(size_Y/2), 
        c_Z-int(size_Z/2):c_Z+int(size_Z/2)]

plt.imshow(X_s[:,:,20])
plt.imshow(Y_s[:,:,20])

# Generate Patches
patch_slices = patch_rows = patch_cols = 32
stride_slices = stride_rows = stride_cols = 16

# WORKAROUND because tensorflow moves the X and y volumes relative to each 
# other if X are int values and Y are float values -> Normalize and Scale the 
# target data with datatype int
Y_s, max_val, min_val = impro.normalize_data(Y_s)
Y_s = impro.scale_data(Y_s, 65535)

# Extract the image patches
session = tf.Session()
p_X = impro.gen_patches2(session=session, data=X_s, patch_slices=patch_slices, 
                   patch_rows=patch_rows, patch_cols=patch_cols, 
                   stride_slices=stride_slices, stride_rows=stride_rows, 
                   stride_cols=stride_cols, input_dim_order='XYZ', 
                   padding='SAME')
p_Y = impro.gen_patches2(session=session, data=Y_s, patch_slices=patch_slices, 
                   patch_rows=patch_rows, patch_cols=patch_cols, 
                   stride_slices=stride_slices, stride_rows=stride_rows, 
                   stride_cols=stride_cols, input_dim_order='XYZ', 
                   padding='SAME')

# WORKAROUND undo the normalization and scaling
p_Y = impro.unscale_data(p_Y, 65535)
p_Y = impro.unnormalize_data(p_Y, max_val, min_val)

plt.imshow(p_X[7,0,12,8,:,:])
plt.imshow(p_Y[7,0,12,8,:,:])

export_path = os.path.join(os.getcwd(), 'mini_dataset')
save_dataset(p_X, p_Y, export_path)

