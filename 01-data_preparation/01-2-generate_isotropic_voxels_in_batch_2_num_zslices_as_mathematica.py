import sys
sys.path.append("..")
import os
import nrrd
import numpy as np
import tensorflow as tf
import javabridge
import bioformats
import sys
from tools import image_processing as impro
from tools import image_io as bfio
import SimpleITK as sitk

# Start the Java VM
javabridge.start_vm(class_path=bioformats.JARS)


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
    
def get_files_in_directory(a_dir):
    files = [f for f in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, f))]
    return files

path_to_data = os.path.join('..', '..', '..', 'Daten2')
subdirs1 = get_immediate_subdirectories(path_to_data)

interpolator = 'bspline'

for subdir1 in subdirs1:
    print('-------------')
    # Spheroid type level
    spheroid_dir = os.path.join(path_to_data, subdir1)
    spheroid_files = get_files_in_directory(spheroid_dir)
    for spheroid_file in spheroid_files:
        if spheroid_file.endswith('.tif'):
            spheroid_name = os.path.splitext(spheroid_file)[0]
            spheroid_file = os.path.join(spheroid_dir, spheroid_file)
            print('Current Spheroid: ', os.path.abspath(spheroid_file))
            subdirs2 = get_immediate_subdirectories(spheroid_dir)
            for subdir2 in subdirs2:
                seg_files = get_files_in_directory(os.path.abspath(os.path.join(spheroid_dir, subdir2)))
                for seg_file in seg_files:
                    if spheroid_name in seg_file and 'NucleiBinary' in seg_file and seg_file.endswith('.tif'):
                        spheroid_file = os.path.abspath(spheroid_file)
                        seg_file = os.path.join(os.path.abspath(spheroid_dir), subdir2, seg_file)
                        print('Corresponding Segmentation: ', seg_file)
                        
                        # Get a numpy array from the tif stack with the dimension
                        spheroid_meta_data, spheroid_data = bfio.get_tif_stack(filepath=spheroid_file, series=0, depth='z', return_dim_order='XYZC') # XYZC
                        segmentation_meta_data, segmentation_data = bfio.get_tif_stack(filepath=seg_file, series=0, depth='t', return_dim_order='XYZC') # XYZC
                        
                        # Transpose the numpy array from XYZC to CZYX for the use with SimpleITK
                        spheroid_data = np.transpose(spheroid_data, axes=[3,2,1,0]) # CZYX
                        
                        # Extract the channel -> make for each channel
                        spheroid_data = spheroid_data[0,:,:,:]
                        
                        # Make a SimpleITK out of the numpy array and set its metadata
                        image = sitk.GetImageFromArray(spheroid_data, isVector=False) # XYZ
                        image.SetOrigin([0.0, 0.0, 0.0])
                        image.SetDirection(np.identity(3, dtype=np.double).flatten().tolist())
                        image.SetSpacing((spheroid_meta_data.get('physical_size_x'), 
                                          spheroid_meta_data.get('physical_size_y'), 
                                          spheroid_meta_data.get('physical_size_z')))
                        
                        # Extract the number of z-slices from the isotropic segmentation
                        num_z_slices = segmentation_data.shape[2]
                        
                        resampled_image = impro.make_image_isotropic(image, interpolator, 0, num_z_slices)
                        
                        # Get a numpy array from the resampled simpleITK image
                        np_image = sitk.GetArrayFromImage(resampled_image)
                        
                        # Transpose the numpy array from ZYX back to to XYZ
                        np_image = np.transpose(np_image, axes=[2,1,0]) # XYZ
                        np_image = np_image.astype('uint8')
                        
                        new_spacing = resampled_image.GetSpacing()
                        header = {"spacings": [new_spacing[0], new_spacing[1], new_spacing[2]], 
                                  "dimension": np_image.ndim,
                                  "type": "uchar", 
                                  "sizes": [resampled_image.GetWidth(), resampled_image.GetHeight(), 
                                            resampled_image.GetDepth()],
                                  "units": ['"microns"', '"microns"', '"microns"']}
                        
                        # Save the resampled image as NRRD-file
                        new_filename = os.path.join(spheroid_dir, spheroid_name)
                        new_filename = new_filename+'.nrrd'
                        nrrd.write(new_filename, data=np_image, header=header, index_order='F')