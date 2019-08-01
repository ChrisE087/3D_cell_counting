import nrrd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import sys
sys.path.append("..")
from tools import image_processing as impro

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
    
def get_files_in_directory(a_dir):
    files = [f for f in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, f))]
    return files

def divide_data(data, dim_order='XYZ'):
    if dim_order == 'ZYX':
        data = np.copy(data)
        data = np.transpose(data, axes=(2,1,0))#XYZ
    x_start = 0
    x_middle = int(np.floor(data.shape[0]/2))
    x_end = data.shape[0]
    west = data[x_start:x_middle, 0:, 0:]
    east = data[x_middle+1:x_end, 0:, 0:]
    return west, east
    

# Specify the path to the input dataset
path_to_data = os.path.join('..', '..', '..', 'Daten')

# Make a list of which spheroids from the dataset are chosen for the training 
# and validation dataset
dataset_list = [['24h', 'C1-untreated_1.1'],
                ['24h', 'C2-untreated_1.1'],
                ['24h', 'C1-untreated_3'],
                ['24h', 'C2-untreated_3'],
                ['48h', 'C1-untreated_1'],
                ['48h', 'C2-untreated_1'],
                ['48h', 'C1-untreated_4.1'],
                ['48h', 'C2-untreated_4.1'],
                ['72h', 'C1-untreated_1'],
                ['72h', 'C2-untreated_1'],
                ['72h', 'C1-untreated_4'],
                ['72h', 'C2-untreated_4']]
#dataset_list = [['24h', 'C1-untreated_3']]

# Specify the path where the generated dataset is saved
train_export_path = os.path.join('dataset', 'train')
val_export_path = os.path.join('dataset', 'val')

# Specify the size of each patch
size_z = 32
size_y = 32
size_x = 32
stride_z = 32
stride_y = 32
stride_x = 32

# padding='SAME' moves the position of the density-map relative to the spheroid
# (Tensorflow-Bug?). When using 'SAME' padding, a workaround is calculated.
# Because of that the generating of the image patches takes much more time! 
# Hence use padding='VALID' whenever possible!
padding = 'VALID'

# Specify the threshold for saving a patch. Patches in which the number of cells
# is greater than the threshold are saved to disk.
thresh = 0.0

# Create the folder if it does not exist
if not os.path.exists(train_export_path):
    try:
        os.makedirs(train_export_path)
    except OSError as e:
        if e.errno != e.errno.EEXIST:
            raise
            
if not os.path.exists(val_export_path):
    try:
        os.makedirs(val_export_path)
    except OSError as e:
        if e.errno != e.errno.EEXIST:
            raise


session = tf.Session()
subdirs1 = get_immediate_subdirectories(path_to_data)
for subdir1 in subdirs1:
    if '24h' in subdir1 or '48h' in subdir1 or '72h' in subdir1:
        # Cultivation period level
        print('-------------')
        data_dir = os.path.join(path_to_data, subdir1)
        subdirs2 = get_immediate_subdirectories(data_dir)
        for subdir2 in subdirs2:
            # Spheroid data level
            untreated_dir = os.path.join(data_dir, subdir2)
            subdirs3 = get_immediate_subdirectories(untreated_dir)
            spheroid_files = get_files_in_directory(untreated_dir)
            # Get all spheroid files in the current directory
            for spheroid in spheroid_files:
                spheroid_name, ext = os.path.splitext(spheroid)
                spheroids_file = os.path.join(untreated_dir, spheroid)
                if ext == '.nrrd':
                    for n in range(len(dataset_list)):
                        if dataset_list[n][0] in spheroids_file and dataset_list[n][1] in spheroids_file:
                            print('Generating Patches for Spheroid: ', spheroids_file)
                            for subdir3 in subdirs3:
                                # OpensegSPIM results level
                                if spheroid_name in subdir3:
                                    result_dir = os.path.join(untreated_dir, subdir3)
                                    centroids_file = os.path.join(result_dir, 'gauss_centroids_fiji_seg.nrrd')
                                    print('Corresponding Centroid: ', centroids_file)
                                    
                                    split = spheroids_file.split(sep=os.path.sep)
                                    time_range = split[4]
                                    
                                    # Read the data
                                    print('Reading the data...')
                                    X, X_header = nrrd.read(spheroids_file) #XYZ
                                    Y, Y_header = nrrd.read(centroids_file) #XYZ
                                    
                                    X_train, X_val = divide_data(X)
                                    Y_train, Y_val = divide_data(Y)
                                    
                                    # WORKAROUND: Scale the target data (Normalize and
                                    # scale with factor) and make a unit16
                                    # dataset, because when using float32 or float64 the
                                    # tensorflow method extract_volume_patches()
                                    # moves the input and target patches or volumes
                                    # relative to each other
                                    if padding == 'SAME':
                                        print('Calculating the workaround...')
                                        Y_train, max_train, min_train = impro.normalize_data(Y_train)
                                        Y_train = impro.scale_data(Y_train, 65535)
                                        
                                        Y_val, max_val, min_val = impro.normalize_data(Y_val)
                                        Y_val = impro.scale_data(Y_val, 65535)
                                    
                                    print('Generating image patches for input data...')
                                    # Generate the image patches of dimension-order ZYX.
                                    # With the parameter 'input_dim_order='XYZ' the
                                    # input data 'X' is transposed to 'ZYX' before generating
                                    # the patches. So the patches are in dimension-order 'ZYX'
                                    patches_X_train = impro.gen_patches(session=session, data=X_train, patch_slices=size_z, patch_rows=size_y,
                                                            patch_cols=size_x, stride_slices=stride_z, stride_rows=stride_y,
                                                            stride_cols=stride_x, input_dim_order='XYZ', padding=padding) #ZYX
                                    
                                    patches_X_val = impro.gen_patches(session=session, data=X_val, patch_slices=size_z, patch_rows=size_y,
                                                            patch_cols=size_x, stride_slices=stride_z, stride_rows=stride_y,
                                                            stride_cols=stride_x, input_dim_order='XYZ', padding=padding) #ZYX
                                    
                                    print('Generating image patches for target data...')
                                    patches_Y_train = impro.gen_patches(session=session, data=Y_train, patch_slices=size_z, patch_rows=size_y,
                                                            patch_cols=size_x, stride_slices=stride_z, stride_rows=stride_y,
                                                            stride_cols=stride_x, input_dim_order='XYZ', padding=padding) #ZYX
                                    
                                    patches_Y_val = impro.gen_patches(session=session, data=Y_val, patch_slices=size_z, patch_rows=size_y,
                                                            patch_cols=size_x, stride_slices=stride_z, stride_rows=stride_y,
                                                            stride_cols=stride_x, input_dim_order='XYZ', padding=padding) #ZYX
                                    
                                    # Unscale the data and make a float32 dataset again
                                    if padding == 'SAME':
                                        print('Undo the workaround...')
                                        patches_Y_train = impro.unscale_data(patches_Y_train, 65535)
                                        patches_Y_train = impro.unnormalize_data(patches_Y_train, max_train, min_train)
                                        
                                        patches_Y_val = impro.unscale_data(patches_Y_val, 65535)
                                        patches_Y_val = impro.unnormalize_data(patches_Y_val, max_val, min_val)
                                    
                                    print('Saving training patches for ', spheroid, '...')
                                    # Create the training dataset
                                    num_patches_train = patches_X_train.shape[0] * patches_X_train.shape[1] * patches_X_train.shape[2]
                                    p_train=0
                                    
                                    print('X-Patches shape = ', patches_X_train.shape)
                                    print('Y-Patches shape = ', patches_Y_train.shape)
                                    if(patches_X_train.shape != patches_Y_train.shape):
                                        print('Warning! Shape of patches for ', spheroids_file, ' ist not equal to shape of patches for ', centroids_file)
            
                                    for pz in range (patches_X_train.shape[0]):
                                        for py in range (patches_X_train.shape[1]):
                                            for px in range (patches_X_train.shape[2]):
                                                if p_train % 100 == 0:
                                                    progress = p_train*100/num_patches_train
                                                    sys.stdout.write('\r' + "{0:.2f}".format(progress) + '%')
                                                    sys.stdout.flush()
                                                #print('Writing patch ', p, ' of ', num_patches)
            
                                                # Read a sample out of the input and target data
                                                patch_X_train = patches_X_train[pz,py,px,:,:,:]
                                                patch_Y_train = patches_Y_train[pz,py,px,:,:,:]
            
                                                # Don't save the paddings
                                                if(np.sum(patch_Y_train) > thresh):
                                                    # Generate a pair of data
                                                    sample_train = np.zeros(shape=(2, patch_X_train.shape[0], patch_X_train.shape[1], patch_X_train.shape[2]), dtype=np.float32)
                                                    sample_train[0] = patch_X_train
                                                    sample_train[1] = patch_Y_train
            
                                                    # Save the input and target data sample
                                                    sample_name_train = "%s_%s_%s-%08d.nrrd" % ('train', time_range, spheroid_name, p_train)
                                                    out_file_train = os.path.join(train_export_path, sample_name_train)
                                                    header_train = {"sum": np.sum(patch_Y_train), "pz": pz, "py": py, "px": px, "spacings": X_header.get('spacings'), "units": X_header.get('units')} # Add the scale factors to the header
                                                    nrrd.write(out_file_train, data=sample_train, header=header_train)
                                                p_train = p_train+1
                                                
                                    print('Saving validation patches for ', spheroid, '...')
                                    # Create the training dataset
                                    num_patches_val = patches_X_val.shape[0] * patches_X_val.shape[1] * patches_X_val.shape[2]
                                    p_val=0
                                    
                                    print('X-Patches shape = ', patches_X_val.shape)
                                    print('Y-Patches shape = ', patches_Y_val.shape)
                                    if(patches_X_val.shape != patches_Y_val.shape):
                                        print('Warning! Shape of patches for ', spheroids_file, ' ist not equal to shape of patches for ', centroids_file)
            
                                    for pz in range (patches_X_val.shape[0]):
                                        for py in range (patches_X_val.shape[1]):
                                            for px in range (patches_X_val.shape[2]):
                                                if p_val % 100 == 0:
                                                    progress = p_val*100/num_patches_val
                                                    sys.stdout.write('\r' + "{0:.2f}".format(progress) + '%')
                                                    sys.stdout.flush()
                                                #print('Writing patch ', p_val, ' of ', num_patches_val)
            
                                                # Read a sample out of the input and target data
                                                patch_X_val = patches_X_val[pz,py,px,:,:,:]
                                                patch_Y_val = patches_Y_val[pz,py,px,:,:,:]
            
                                                # Don't save the paddings
                                                if(np.sum(patch_Y_val) > thresh):
                                                    # Generate a pair of data
                                                    sample_val = np.zeros(shape=(2, patch_X_val.shape[0], patch_X_val.shape[1], patch_X_val.shape[2]), dtype=np.float32)
                                                    sample_val[0] = patch_X_val
                                                    sample_val[1] = patch_Y_val
            
                                                    # Save the input and target data sample
                                                    sample_name_val = "%s_%s_%s-%08d.nrrd" % ('val', time_range, spheroid_name, p_val)
                                                    out_file_val = os.path.join(val_export_path, sample_name_val)
                                                    header_val = {"sum": np.sum(patch_Y_val), "pz": pz, "py": py, "px": px, "spacings": X_header.get('spacings'), "units": X_header.get('units')} # Add the scale factors to the header
                                                    nrrd.write(out_file_val, data=sample_val, header=header_val)
                                                p_val = p_val+1