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
path_to_data = os.path.join('..', '..', '..', 'Daten2')

# Specify the suffix of the target-data
target_suffix = 'NucleiBinary' # 'gauss_centroids' or 'NucleiBinary'

# Make a list of which spheroids from the dataset are chosen for the training 
# and validation dataset
train_list = [['Fibroblasten', '1_draq5'],
              ['Fibroblasten', '4_draq5'],
              #['Fibroblasten', '8_draq5'],
              ['Fibroblasten', '9_draq5'],
              #['Hacat', 'C3-2'],
              ['Hacat', 'C3-3'],
              #['Hacat', 'C3-8'],
              ['Hacat', 'C3-9'],
              ['HT29', 'C2-HT29_Glycerol_Ki67_02'],
              ['HT29', 'C2-HT29_Glycerol_Ki67_04'],
              #['HT29', 'C2-HT29_Glycerol_Ki67_06'],
              ['HT29', 'C2-HT29_Glycerol_Ki67_09'],
              ['HTC8', 'C3-3'],
              ['HTC8', 'C3-5'],
              #['HTC8', 'C3-6r'],
              ['HTC8', 'C3-10'],
              ['NPC1', 'C3-2'],
              ['NPC1', 'C3-5'],
              #['NPC1', 'C3-6'],
              ['NPC1', 'C3-9']]

val_list = [['Fibroblasten', '2_draq5'],
            ['Hacat', 'C3-4'],
            ['HT29', 'C2-HT29_Glycerol_Ki67_03'],
            ['HTC8', 'C3-2a'],
            ['NPC1', 'C3-4']]

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
                for m in range(len(data_list)):
                    if data_list[m][0] in spheroid_file and data_list[m][1] in spheroid_file:
                        print('Generating Patches for Spheroid: ', spheroid_file)
                        subdirs2 = get_immediate_subdirectories(spheroid_dir)
                        for subdir2 in subdirs2:
                            res_dir = os.path.abspath(os.path.join(spheroid_dir, subdir2))
                            res_files = get_files_in_directory(res_dir)
                            for res_file in res_files:
                                if spheroid_name + '-' + target_suffix in res_file and res_file.endswith('.nrrd'):
                                    spheroid_file = os.path.abspath(spheroid_file)
                                    target_file = os.path.join(os.path.abspath(spheroid_dir), subdir2, res_file)
                                    
                                    spheroid_category = spheroid_file.split(sep=os.path.sep)
                                    spheroid_category = spheroid_category[2]
                                                
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
                                    
                                    if padding == 'SAME':
                                        print('Undo the workaround...')
                                        patches_y = impro.unscale_data(patches_y, 65535)
                                        patches_y = impro.unnormalize_data(patches_y, y_max, y_min)
                                    
                                    # Create the training dataset
                                    print('Saving patches for ', spheroid_name, '...')
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
                                                
                                                # Make the segmentation binary and bring it into range 0 to 1
                                                if target_suffix == 'NucleiBinary':
                                                    patch_y[patch_y > 0] = 1
            
                                                # Don't save the paddings
                                                if(np.sum(patch_y) > thresh):
                                                    # Generate a pair of data
                                                    sample = np.zeros(shape=(2, patch_X.shape[0], patch_X.shape[1], patch_X.shape[2]), dtype=np.float32)
                                                    sample[0] = patch_X
                                                    sample[1] = patch_y
                                                    
                                                    # Generate the export path and filename
                                                    if n == 0:
                                                        sample_name = "%s_%s_%s-%08d.nrrd" % ('train', spheroid_category, spheroid_name, p)
                                                        export_path = train_export_path
                                                    else:
                                                        sample_name = "%s_%s_%s-%08d.nrrd" % ('val', spheroid_category, spheroid_name, p)
                                                        export_path = val_export_path
            
                                                    # Save the input and target data sample
                                                    out_file = os.path.join(export_path, sample_name)
                                                    out_header = {"sum": np.sum(patch_y), "pz": pz, "py": py, "px": px, "spacings": X_header.get('spacings'), "units": X_header.get('units')} # Add the scale factors to the header
                                                    nrrd.write(out_file, data=sample, header=out_header)
                                                p = p+1