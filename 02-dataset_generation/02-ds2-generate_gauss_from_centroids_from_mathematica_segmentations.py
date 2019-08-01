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

table = []
table.append(['Spheroid', 'Number of cells (sum over density-map)', 'Min value', 'Max value'])

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
                seg_files = get_files_in_directory(res_dir)
                for seg_file in seg_files:
                    if spheroid_name in seg_file and 'centroids' in seg_file and seg_file.endswith('.nrrd'):
                        spheroid_file = os.path.abspath(spheroid_file)
                        centroids_file = os.path.join(os.path.abspath(spheroid_dir), subdir2, seg_file)
                        print('Corresponding Centroids: ', centroids_file)
                        centroids, centroids_header = nrrd.read(centroids_file) # XYZ
                        gauss_centroids = impro.convolve_with_gauss(centroids, 50, 6)
                        nrrd_gauss_centroids_file = os.path.join(res_dir, spheroid_name+'-gauss_centroids.nrrd')
                        nrrd.write(nrrd_gauss_centroids_file, data=gauss_centroids, header=centroids_header, index_order='F')
                        # Log the number of cells, min and max in a table
                        spheroid_title = res_dir.split(os.path.sep)[2] + '->' + spheroid_name
                        table.append([spheroid_title, np.sum(gauss_centroids), np.min(gauss_centroids), np.max(gauss_centroids)])
                        
with open('gauss_cell_numbers_dataset2_mathematica_segmentations_filtered.txt','w') as file:
    for item in table:
        line = "%s \t %s \t %s \t %s\n" %(item[0], item[1], item[2], item[3])
        file.write(line)