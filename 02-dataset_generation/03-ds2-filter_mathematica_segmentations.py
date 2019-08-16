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
#javabridge.start_vm(class_path=bioformats.JARS)


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
    
def get_files_in_directory(a_dir):
    files = [f for f in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, f))]
    return files

path_to_data = os.path.join('..', '..', '..', 'Daten2')
subdirs1 = get_immediate_subdirectories(path_to_data)

table = []
table.append(['Spheroid', 'Number of cells befor filtering (max label)', 'Number of cells befor filtering (max label)'])

# Specify the excluded volume size of the segmentations that are set to zero
excluded_volume_size = 3 #um^3

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
            subdirs2 = get_immediate_subdirectories(spheroid_dir)
            for subdir2 in subdirs2:
                res_dir = os.path.abspath(os.path.join(spheroid_dir, subdir2))
                files = get_files_in_directory(res_dir)
                for file in files:
                    if spheroid_name + '-NucleiBinary' in file and file.endswith('.nrrd'):
                        spheroid_file = os.path.abspath(spheroid_file)
                        segmentation_file = os.path.join(os.path.abspath(spheroid_dir), subdir2, file)
                        print('Corresponding Centroids: ', segmentation_file)
                        segmentation, header = nrrd.read(segmentation_file) # XYZ
                        spacings = header.get('spacings')
                        print(spacings)
                        segmentation_filtered = impro.filter_segmentation(segmentation, spacings, excluded_volume_size)
                        filtered_segmentation_file = os.path.join(res_dir, spheroid_name+'-NucleiBinary_filtered.nrrd')
                        nrrd.write(filtered_segmentation_file, data=segmentation_filtered, header=header, index_order='F')
                        # Log the number of cells, min and max in a table
                        spheroid_title = res_dir.split(os.path.sep)[2] + '->' + spheroid_name
                        table.append([spheroid_title, np.max(segmentation), np.max(segmentation_filtered)])
                        
with open('cell_numbers_dataset2_mathematica_segmentations_filtered.txt','w') as file:
    for item in table:
        line = "%s \t %s \t %s\n" %(item[0], item[1], item[2])
        file.write(line)