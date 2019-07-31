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
import cc3d

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

for subdir1 in subdirs1:
    print('-------------')
    # Spheroid type level
    spheroid_dir = os.path.join(path_to_data, subdir1)
    spheroid_files = get_files_in_directory(spheroid_dir)
    for spheroid_file in spheroid_files:
        if spheroid_file.endswith('.nrrd'):
            spheroid_name = os.path.splitext(spheroid_file)[0]
            spheroid_file = os.path.join(spheroid_dir, spheroid_file)
            print('Current Spheroid: ', os.path.abspath(spheroid_file))
            subdirs2 = get_immediate_subdirectories(spheroid_dir)
            for subdir2 in subdirs2:
                seg_files = get_files_in_directory(os.path.abspath(os.path.join(spheroid_dir, subdir2)))
                for seg_file in seg_files:
                    if spheroid_name in seg_file and 'NucleiBinary' in seg_file and seg_file.endswith('.tif'):
                        spheroid_file = os.path.abspath(spheroid_file)
                        spheroid_dir = os.path.abspath(spheroid_dir)
                        seg_file = os.path.join(spheroid_dir, subdir2, seg_file)
                        seg_filename = os.path.splitext(seg_file)[0]
                        print('Corresponding Segmentation: ', seg_file)
                        
                        # Load the Spheroid and corresponding segmentation
                        spheroid_data, spheroid_header = nrrd.read(spheroid_file)
                        segmentation_meta_data, segmentation_data = bfio.get_tif_stack(filepath=seg_file, series=0, depth='t', return_dim_order='XYZC') # XYZC
                        
                        segmentation_data = segmentation_data[:,:,:,0]
                        segmentation_data = np.transpose(segmentation_data, axes=(2,1,0)) # ZYX
                        
                        # Label the segmentations
                        labelled_segmentation_data = cc3d.connected_components(segmentation_data, connectivity=6)
                        
                        labelled_segmentation_data = np.transpose(labelled_segmentation_data, axes=(2,1,0)).astype(np.uint16) # XYZ
                        
                        # Save the labelled data as nrrd file
                        nrrd.write(os.path.join(spheroid_dir, subdir2, seg_filename+'.nrrd'), data=labelled_segmentation_data, header=spheroid_header, index_order='F')