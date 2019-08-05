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
path_to_data = os.path.join('..', '..', '..', 'Daten2')

# Make a list of which spheroids from the dataset are chosen for the training 
# and validation dataset
dataset_list = [['Fibroblasten', '1_draq5'],
                ['Fibroblasten', '9_draq5'],
                ['Hacat', 'C3-3'],
                ['Hacat', 'C3-8'],
                ['HT29', 'C2-HT29_Glycerol_Ki67_02'],
                ['HT29', 'C2-HT29_Glycerol_Ki67_04'],
                ['HTC8', 'C3-3'],
                ['HTC8', 'C3-5'],
                ['NPC1', 'C3-5'],
                ['NPC1', 'C3-9'],
                ['HT29', 'C2-HT29_Glycerol_Ki67_06'],
                ['HT29', 'C2-HT29_Glycerol_Ki67_09'],
                ['HTC8', 'C3-6r'],
                ['HTC8', 'C3-10'],
                ['Fibroblasten', '4_draq5'],
                ['Fibroblasten', '8_draq5'],
                ['Hacat', 'C3-2'],
                ['Hacat', 'C3-9'],
                ['NPC1', 'C3-2'],
                ['NPC1', 'C3-10']]


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
    print('-------------')
    # Spheroid type level
    spheroid_dir = os.path.join(path_to_data, subdir1)
    spheroid_files = get_files_in_directory(spheroid_dir)
    for spheroid_file in spheroid_files:
        if spheroid_file.endswith('.nrrd'):
            print('Processing files:')
            spheroid_name = os.path.splitext(spheroid_file)[0]
            spheroid_file = os.path.join(spheroid_dir, spheroid_file)
            print('Current Spheroid: ', os.path.abspath(spheroid_file))
            for n in range(len(dataset_list)):
                if dataset_list[n][0] in spheroid_file and dataset_list[n][1] in spheroid_file:
                    print('Generating Patches for Spheroid: ', spheroid_file)
                    subdirs2 = get_immediate_subdirectories(spheroid_dir)
                    for subdir2 in subdirs2:
                        res_dir = os.path.abspath(os.path.join(spheroid_dir, subdir2))
                        seg_files = get_files_in_directory(res_dir)
                        for seg_file in seg_files:
                            if spheroid_name + '-gauss_centroids' in seg_file and seg_file.endswith('.nrrd'):
                                spheroids_file = os.path.abspath(spheroid_file)
                                centroids_file = os.path.join(os.path.abspath(spheroid_dir), subdir2, seg_file)
                                
                                spheroid_category = spheroids_file.split(sep=os.path.sep)
                                spheroid_category = spheroid_category[2]
                                            
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
                                
                                if padding == 'SAME':
                                    # Unscale the data and make a float32 dataset again
                                    print('Undo the workaround...')
                                    patches_Y_train = impro.unscale_data(patches_Y_train, 65535)
                                    patches_Y_train = impro.unnormalize_data(patches_Y_train, max_train, min_train)
                                    
                                    patches_Y_val = impro.unscale_data(patches_Y_val, 65535)
                                    patches_Y_val = impro.unnormalize_data(patches_Y_val, max_val, min_val)
                                
                                print('Saving training patches for ', spheroid_name, '...')
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
                                                sample_name_train = "%s_%s_%s-%08d.nrrd" % ('train', spheroid_category, spheroid_name, p_train)
                                                out_file_train = os.path.join(train_export_path, sample_name_train)
                                                header_train = {"sum": np.sum(patch_Y_train), "pz": pz, "py": py, "px": px, "spacings": X_header.get('spacings'), "units": X_header.get('units')} # Add the scale factors to the header
                                                nrrd.write(out_file_train, data=sample_train, header=header_train)
                                            p_train = p_train+1
                                            
                                print('Saving validation patches for ', spheroid_name, '...')
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
                                                sample_name_val = "%s_%s_%s-%08d.nrrd" % ('val', spheroid_category, spheroid_name, p_val)
                                                out_file_val = os.path.join(val_export_path, sample_name_val)
                                                header_val = {"sum": np.sum(patch_Y_val), "pz": pz, "py": py, "px": px, "spacings": X_header.get('spacings'), "units": X_header.get('units')} # Add the scale factors to the header
                                                nrrd.write(out_file_val, data=sample_val, header=header_val)
                                            p_val = p_val+1