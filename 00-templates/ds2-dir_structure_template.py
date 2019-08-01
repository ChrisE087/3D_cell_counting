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