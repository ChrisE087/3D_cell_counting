import numpy as np
import tensorflow as tf
import nrrd
import os
import matplotlib.pyplot as plt

def gen_patches(data, patch_slices, patch_rows, patch_cols, stride_slices, 
                stride_rows, stride_cols, input_dim_order='XYZ', padding='VALID'):
    
    # Reorder the dimensions to ZYX
    if input_dim_order == 'XYZ':
        data = np.transpose(data, axes=(2,1,0))
    
    # Check if the data has channels
    if np.size(data.shape) != 3:
        print('WARNING! Function is only meant to be used for data with one channel')
        return
    
    # Expand dimension for depth (number of channels)
    data = data[:,:,:,np.newaxis]
    
    # Expand the dimension for batches
    data = data[np.newaxis,:,:,:,:]
    
    # Extract  patches of size patch_slices x patch_rows x patch_cols
    with tf.Session() as sess:
        t = tf.extract_volume_patches(data, ksizes=[1, patch_slices, patch_rows, patch_cols, 1], 
                                      strides=[1, stride_slices, stride_rows, stride_cols, 1], 
                                      padding=padding).eval(session=sess)
    
        # Reshape the patches to 3D
        # t.shape[1] -> number of extracted patches in z-direction
        # t.shape[2] -> number of extracted patches in y-direction
        # t.shape[3] -> number of extracted patches in x-direction
        t = tf.reshape(t, [1, t.shape[1], t.shape[2], t.shape[3], 
                           patch_slices, patch_rows, patch_cols]).eval(session=sess)
        sess.close()
    
    # Remove the batch dimension
    patches = t[0,:,:,:,:]
    
    # Remove the channel dimension
    #if has_channels == False:
        #patches = t[:,:,:,0]
    
    return patches

def restore_volume(patches, output_dim_order='XYZ'):
    
    # Extract the patches and build the volume
    for zslice in range(patches.shape[0]):
        for row in range(patches.shape[1]):
            for col in range(patches.shape[2]):
                # Extract a 3D-patch
                patch = patches[zslice, row, col, :]
                
                # First column-patch? -> Initialize a volume, else concatenate with the
                # last patch on the column-axis
                if(col == 0):
                    col_concat = patch
                else:
                    col_concat = np.concatenate((col_concat, patch), axis=2)
            
            # First row-patch? -> Initialize a volume, else concatenate with the
            # last patch on the row-axis
            if(row == 0):
                row_concat = col_concat
            else:
                row_concat = np.concatenate((row_concat, col_concat), axis=1)
        # First slice-patch? -> Initialize a volume, else concatenate with the
        # last patch on the slice-axis
        if(zslice == 0):
            slice_concat = row_concat
        else:
            slice_concat = np.concatenate((slice_concat, row_concat), axis=0)

    # The output volume is the over all three axes concatenated patches
    if output_dim_order == 'ZYX':
        return slice_concat
    if output_dim_order == 'XYZ':
        return np.transpose(slice_concat, axes=(2,1,0))


path_X = os.path.join('test_data', 'test_input_data.nrrd')
path_y = os.path.join('test_data', 'test_output_data.nrrd')

# Load the volumes
X, X_header = nrrd.read(path_X) # XYZ
y, y_header = nrrd.read(path_y) # XYZ

# Specify the size of each patch
size_z = 64
size_y = 64
size_x = 64

patches_X = gen_patches(data=X, patch_slices=size_z, patch_rows=size_y, 
                      patch_cols=size_x, stride_slices=size_z, stride_rows=size_y, 
                      stride_cols=size_x, input_dim_order='XYZ', padding='VALID')

patches_y = gen_patches(data=y, patch_slices=size_z, patch_rows=size_y, 
                      patch_cols=size_x, stride_slices=size_z, stride_rows=size_y, 
                      stride_cols=size_x, input_dim_order='XYZ', padding='VALID')



# Restore the volume
restored_X = restore_volume(patches_X, output_dim_order='XYZ')
restored_y = restore_volume(patches_y, output_dim_order='XYZ')



plt.imshow(restored_X[:,:,50])
plt.imshow(restored_y[:,:,50])

nrrd.write('restoredX.nrrd', restored_X)
nrrd.write('restoredy.nrrd', restored_y)


# Create the dataset 
num_patches = patches_X.shape[0] * patches_X.shape[1] * patches_X.shape[2]
X_data = np.zeros(shape=(num_patches, size_z, size_y, size_x), dtype='uint8')
y_data = np.zeros(shape=(num_patches, size_z, size_y, size_x), dtype='float64')

patch_num = 0
for a in range (patches_X.shape[0]):
    for b in range (patches_X.shape[1]):
        for c in range (patches_X.shape[2]):
            print(patch_num)
            X_data[patch_num,:,:,:] = patches_X[a, b, c,:,:,:]
            y_data[patch_num,:,:,:] = patches_y[a, b, c,:,:,:]
            patch_num += 1
            
plt.imshow(y_data[56,16,:,:])
plt.imshow(X_data[56,16,:,:])

# Transpose the dataset to BatchSize and XYZ
X_data = np.transpose(X_data, axes=(0,3,2,1))
y_data = np.transpose(y_data, axes=(0,3,2,1))

nrrd.write('X.nrrd', X_data)
nrrd.write('y.nrrd', y_data)
