import sys
sys.path.append("..")
import os
import numpy as np
import nrrd
from tools import image_io as bfio
import javabridge
import bioformats

# Start the Java VM
#javabridge.start_vm(class_path=bioformats.JARS)

path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', '..', 'Datensaetze', 'Aufnahmen_und_Segmentierungen', 'Datensatz1'))

table = []
table.append(['Cultivation Period', 'Spheroid', 'Dimension (non isotropic)', 'Dimension (isotropic)'])

for directory in os.listdir(path_to_data):
    print(directory)
    if directory == '24h' or directory == '48h' or directory == '72h':
        print('###########################################################')
        cultivation_period = directory
        data_dir = os.path.join(path_to_data, directory, 'untreated')
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.nrrd'):
                    spheroid_name = os.path.splitext(filename)[0]
                    print('Actual file: ', spheroid_name)
                    
                    # Read the original data
                    tif_file = os.path.join(data_dir, spheroid_name+'.tif')
                    tif_header, tif_data = bfio.get_tif_stack(filepath=tif_file, series=0, depth='z', return_dim_order='XYZC') # XYZC
                    tif_data = tif_data[:,:,:,0]
                    tif_data = np.transpose(tif_data, axes=(2,1,0)) #ZYX
                    print('TIF Dimension: ', tif_data.shape)
                    
                    # Read the isotropic data
                    nrrd_file = os.path.join(data_dir, spheroid_name+'.nrrd')
                    nrrd_data, nrrd_header = nrrd.read(nrrd_file)
                    nrrd_data = np.transpose(nrrd_data, axes=(2,1,0)) #ZYX
                    print('NRRD Dimension: ', nrrd_data.shape)
                    
                    # Save the dimensions in a table
                    table.append([cultivation_period, spheroid_name, tif_data.shape[0:3], nrrd_data.shape])
                
with open('image_dimensions.txt','w') as file:
    for item in table:
        line = "%s \t %s \t %s \t %s\n" %(item[0], item[1], item[2], item[3])
        file.write(line)