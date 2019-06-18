import sys
sys.path.append("..")
import os
import nrrd
import numpy as np
from tools import image_processing as impro


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
    
def get_files_in_directory(a_dir):
    files = [f for f in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, f))]
    return files

# Specify the size of each patch
size_z = 32
size_y = 32
size_x = 32
stride_z = 24
stride_y = 24
stride_x = 24

# Specify the path to the dataset
path_to_dataset = os.path.join('dataset')

# Create the folder if it does not exist
if not os.path.exists(path_to_dataset):
    try:
        os.makedirs(path_to_dataset)
    except OSError as e:
        if e.errno != e.errno.EEXIST:
            raise

path_to_data = os.path.join('..', '..', '..', 'Daten')
subdirs1 = get_immediate_subdirectories(path_to_data)

for subdir1 in subdirs1:
    if '24h' in subdir1 or '48h' in subdir1 or '72h' in subdir1:
        print('-------------')
        # Cultivation period level
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
                    print('Current Spheroid: ', spheroid)
                    for subdir3 in subdirs3:
                        # OpensegSPIM results level
                        if spheroid_name in subdir3 and not "24h" in subdir1 and not "48h" in subdir1:
                            #if 'C2-untreated_2.2' in spheroid or 'C2-untreated_2.3' in spheroid or 'C2-untreated_3' in spheroid:
                            #if not 'C1-untreated_1.1' in spheroid:
                            result_dir = os.path.join(untreated_dir, subdir3)
                            centroids_file = os.path.join(result_dir, 'gauss_centroids.nrrd')
                            print('Corresponding Centroid: ', centroids_file)
                            
                            split = spheroids_file.split(sep=os.path.sep)
                            time_range = split[4]
                            
                            # Read the data
                            X, X_header = nrrd.read(spheroids_file) #XYZ
                            Y, Y_header = nrrd.read(centroids_file) #XYZ
                            
                            # Scale the target data
                            Y, max_val, min_val = impro.normalize_data(Y)
                            Y = impro.scale_data(Y, 65535)
                            
                            print('Generating image patches for input data...')
                            # Generate the image patches
                            patches_X = impro.gen_patches(data=X, patch_slices=size_z, patch_rows=size_y, 
                                                    patch_cols=size_x, stride_slices=stride_z, stride_rows=stride_y, 
                                                    stride_cols=stride_x, input_dim_order='XYZ', padding='SAME') #ZYX

                            print('Generating image patches for target data...')
                            patches_y = impro.gen_patches(data=Y, patch_slices=size_z, patch_rows=size_y, 
                                                    patch_cols=size_x, stride_slices=stride_z, stride_rows=stride_y, 
                                                    stride_cols=stride_x, input_dim_order='XYZ', padding='SAME') #ZYX
                            
                            # Unscale the target data
                            patches_y = impro.unscale_data(patches_y, 65535)
                            patches_y = impro.unnormalize_data(patches_y, max_val, min_val)
                            
                            # Create the dataset 
                            num_patches = patches_X.shape[0] * patches_X.shape[1] * patches_X.shape[2]
                            p=0
                           
                            for a in range (patches_X.shape[0]):
                                for b in range (patches_X.shape[1]):
                                    for c in range (patches_X.shape[2]):
                                        print('Writing patch ', p, ' of ', num_patches)
                                        
                                        # Read a sample out of the input and target data
                                        patch_x = patches_X[a, b, c,:,:,:]
                                        patch_y = patches_y[a, b, c,:,:,:]
                                        
                                        # Generate a sample
                                        sample = np.zeros(shape=(2, patch_x.shape[0], patch_x.shape[1], patch_x.shape[2]), dtype=np.float32)
                                        sample[0] = patch_x
                                        sample[1] = patch_y
                                        
                                        # Transpose the data back to XYZ
                                        sample = np.transpose(sample, axes=(0,3,2,1))
                                        
                                        # Save the input and target data sample                             
                                        sample_name = "%s_%s-%04d.nrrd" % (time_range, spheroid_name, p)
                                        out_file = os.path.join(path_to_dataset, sample_name)
                                        header = {"max_val": max_val, "min_val": min_val, "spacings": X_header.get('spacings'), "units": X_header.get('units')} # Add the scale factors to the header
                                        nrrd.write(out_file, data=sample, header=header)
                                        
                                        p += 1