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

path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', '..', 'Datensaetze', 'Aufnahmen_und_Segmentierungen', 'Datensatz2'))
subdirs1 = get_immediate_subdirectories(path_to_data)

table = []
table.append(['Cultivation Period', 'Spheroid', 'Dimension (non isotropic)', 'Dimension (isotropic)'])

for subdir1 in subdirs1:
    print('-------------')
    cell_type = subdir1
    print(cell_type)
    # Spheroid type level
    spheroid_dir = os.path.join(path_to_data, subdir1)
    spheroid_files = get_files_in_directory(spheroid_dir)
    for spheroid_file in spheroid_files:
        if spheroid_file.endswith('.nrrd'):
            spheroid_name = os.path.splitext(spheroid_file)[0]
            print('Processing', spheroid_name)
            
            # Read the original data
            tif_file = os.path.join(spheroid_dir, spheroid_name+'.tif')
            tif_header, tif_data = bfio.get_tif_stack(filepath=tif_file, series=0, depth='z', return_dim_order='XYZC') # XYZC
            tif_data = tif_data[:,:,:,0]
            tif_data = np.transpose(tif_data, axes=(2,1,0)) #ZYX
            print('TIF Dimension: ', tif_data.shape)
            
            # Read the isotropic data
            nrrd_file = os.path.join(spheroid_dir, spheroid_name+'.nrrd')
            nrrd_data, nrrd_header = nrrd.read(nrrd_file)
            nrrd_data = np.transpose(nrrd_data, axes=(2,1,0)) #ZYX
            print('NRRD Dimension: ', nrrd_data.shape)
            
            # Save the dimensions in a table
            table.append([cell_type, spheroid_name, tif_data.shape[0:3], nrrd_data.shape])
            
                        
with open('image_dimensions.txt','w') as file:
    for item in table:
        line = "%s \t %s \t %s \t %s\n" %(item[0], item[1], item[2], item[3])
        file.write(line)