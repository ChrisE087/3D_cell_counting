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

#%%############################################################################
# Specify the parameters
###############################################################################    

# Specify the path to the input dataset
path_to_data = os.path.join('..', '..', '..', 'Daten')

# Specify the filename of the target data
target_filename = 'Nucleisegmentedfill2r.nrrd' # 'gauss_centroids_fiji_seg.nrrd' or 'gauss_centroids_opensegspim_seg.nrrd' (not recommended!) for density-patches or 'Nucleisegmentedfill2r.nrrd' for segmentation-patches

# Make a list of which spheroids from the dataset are chosen for the training 
# and validation dataset
train_list = [
              ['24h', 'C2-untreated_2.1'],
              ['48h', 'C2-untreated_3'],
              ['72h', 'C2-untreated_3']]
#train_list = [#['24h', 'C1-untreated_1.1'],
#              ['24h', 'C2-untreated_1.1'],
#              #['24h', 'C1-untreated_2.1'],
#              ['24h', 'C2-untreated_2.1'],
#              #['24h', 'C1-untreated_3'],
#              ['24h', 'C2-untreated_3'],
#              #['48h', 'C1-untreated_1'],
#              ['48h', 'C2-untreated_1'],
#              #['48h', 'C1-untreated_3'],
#              ['48h', 'C2-untreated_3'],
#              #['48h', 'C1-untreated_4.1'],
#              ['48h', 'C2-untreated_4.1'],
#              #['72h', 'C1-untreated_1'],
#              ['72h', 'C2-untreated_1'],
#              #['72h', 'C1-untreated_3'],
#              ['72h', 'C2-untreated_3'],
#              #['72h', 'C1-untreated_4'],
#              ['72h', 'C2-untreated_4']]

val_list = []
#val_list = [#['24h', 'C1-untreated_1.2'],
#            ['24h', 'C2-untreated_1.2'],
#            #['48h', 'C1-untreated_2'],
#            ['48h', 'C2-untreated_2'],
#            #['72h', 'C1-untreated_2'],
#            ['72h', 'C2-untreated_2']]

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

#%%############################################################################
# Create the export folders if they don't exist
###############################################################################
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

#%%############################################################################
# Generate the dataset
###############################################################################
dataset_list = [train_list, val_list]
session = tf.Session()
for n in range(len(dataset_list)):
    data_list = dataset_list[n]
    if n == 0:
        print('########### Generating training data...')
    else:
        print('########### Generating validation data...')
    
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
                    spheroid_file = os.path.join(untreated_dir, spheroid)
                    if ext == '.nrrd':
                        for m in range(len(data_list)):
                            if data_list[m][0] in spheroid_file and data_list[m][1] in spheroid_file:
                                print('Generating Patches for Spheroid: ', spheroid_file)
                                for subdir3 in subdirs3:
                                    # OpensegSPIM results level
                                    if spheroid_name in subdir3:
                                        result_dir = os.path.join(untreated_dir, subdir3)
                                        target_file = os.path.join(result_dir, target_filename)
                                        print('Corresponding target: ', target_file)

                                        split = spheroid_file.split(sep=os.path.sep)
                                        time_range = split[4]
                                        
                                        # Read the data
                                        print('Reading the data...')
                                        X, X_header = nrrd.read(spheroid_file) #XYZ
                                        y, y_header = nrrd.read(target_file) #XYZ
                                        
                                        # WORKAROUND: Scale the target data (Normalize and
                                        # scale with factor) and make a unit16
                                        # dataset, because when using float32 or float64 the
                                        # tensorflow method extract_volume_patches()
                                        # moves the input and target patches or volumes
                                        # relative to each other
                                        if padding == 'SAME':
                                            print('Calculating the workaround...')
                                            y, y_max, y_min = impro.normalize_data(y)
                                            y = impro.scale_data(y, 65535)
                                        
                                        print('Generating image patches for input data...')
                                        # Generate the image patches of dimension-order ZYX.
                                        # With the parameter 'input_dim_order='XYZ' the
                                        # input data 'X' is transposed to 'ZYX' before generating
                                        # the patches. So the patches are in dimension-order 'ZYX'
                                        patches_X = impro.gen_patches(session=session, data=X, patch_slices=size_z, patch_rows=size_y,
                                                                patch_cols=size_x, stride_slices=stride_z, stride_rows=stride_y,
                                                                stride_cols=stride_x, input_dim_order='XYZ', padding=padding) #ZYX
                                        
                                        print('Generating image patches for target data...')
                                        patches_y = impro.gen_patches(session=session, data=y, patch_slices=size_z, patch_rows=size_y,
                                                                patch_cols=size_x, stride_slices=stride_z, stride_rows=stride_y,
                                                                stride_cols=stride_x, input_dim_order='XYZ', padding=padding) #ZYX
                                        
                                        # Unscale the data and make a float32 dataset again
                                        if padding == 'SAME':
                                            print('Undo the workaround...')
                                            patches_y = impro.unscale_data(patches_y, 65535)
                                            patches_y = impro.unnormalize_data(patches_y, y_max, y_min)
                                        
                                        # Create the dataset
                                        print('Saving patches for ', spheroid, '...')
                                        num_patches = patches_X.shape[0] * patches_X.shape[1] * patches_X.shape[2]
                                        p=0
                                        
                                        print('X-Patches shape = ', patches_X.shape)
                                        print('Y-Patches shape = ', patches_y.shape)
                                        if(patches_X.shape != patches_y.shape):
                                            print('Warning! Shape of patches for ', spheroid_file, ' ist not equal to shape of patches for ', target_file)
                
                                        for pz in range (patches_X.shape[0]):
                                            for py in range (patches_X.shape[1]):
                                                for px in range (patches_X.shape[2]):
                                                    if p % 100 == 0:
                                                        progress = p*100/num_patches
                                                        sys.stdout.write('\r' + "{0:.2f}".format(progress) + '%')
                                                        sys.stdout.flush()
                
                                                    # Read a sample out of the input and target data
                                                    patch_X = patches_X[pz,py,px,:,:,:]
                                                    patch_y = patches_y[pz,py,px,:,:,:]
                                                    
                                                    if target_filename == 'Nucleisegmentedfill2r.nrrd' or target_filename == 'Nucleisegmentedfill.nrrd':
                                                        patch_y = patch_y / 255
                
                                                    # Don't save the paddings
                                                    if(np.sum(patch_y) > thresh):
                                                        # Generate a pair of data
                                                        sample = np.zeros(shape=(2, patch_X.shape[0], patch_X.shape[1], patch_X.shape[2]), dtype=np.float32)
                                                        sample[0] = patch_X
                                                        sample[1] = patch_y
                                                        
                                                        # Generate the export path and filename
                                                        if n == 0:
                                                            sample_name = "%s_%s_%s-%08d.nrrd" % ('train', time_range, spheroid_name, p)
                                                            export_path = train_export_path
                                                        else:
                                                            sample_name = "%s_%s_%s-%08d.nrrd" % ('val', time_range, spheroid_name, p)
                                                            export_path = val_export_path
                
                                                        # Save the input and target data sample
                                                        out_file = os.path.join(export_path, sample_name)
                                                        out_header = {"sum": np.sum(patch_y), "pz": pz, "py": py, "px": px, "spacings": X_header.get('spacings'), "units": X_header.get('units')} # Add the scale factors to the header
                                                        nrrd.write(out_file, data=sample, header=out_header)
                                                    p = p+1