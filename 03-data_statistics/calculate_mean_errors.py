import sys
sys.path.append("..")
import os
import javabridge
import bioformats
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import nrrd
from tools import image_io as bfio
from tools import image_processing as impro

path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Daten'))

table = [['Isotropic Original', 'OpenSegSPIM Data', 'Mean Squared Error']]

for directory in os.listdir(path_to_data):
    data_dir = os.path.join(path_to_data, directory, 'untreated')
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.nrrd'):
                name = os.path.splitext(filename)[0]
                for subdir in os.listdir(data_dir):
                    res_dir = os.path.join(data_dir, subdir)
                    if(os.path.isdir(res_dir)):
                        if name in subdir:
                            original_file = os.path.join(data_dir, filename)
                            opensegspim_file = os.path.join(res_dir, 'OriginalStack.nrrd')
                            print('Processing file: ', original_file, 'and', opensegspim_file)
                            
                            # Read the data
                            original_data, original_data_header = nrrd.read(original_file) #XYZ
                            opensegspim_data, opensegspim_header = nrrd.read(opensegspim_file) #XYZ
                            
                            # Convert the 16-bit OpenSegSPIM data to 8-bit
                            opensegspim_data = (opensegspim_data/256).astype('uint8')
                            
                            # Convert the data to float for calculating the error
                            original_data = original_data.astype('float')
                            opensegspim_data = opensegspim_data.astype('float')
                            
                            # Calculate the error volume
                            error = np.abs((original_data - opensegspim_data))
                            
                            # Calculate the mean squared error
                            mse = np.sum(error*error)/np.size(error)
                            
                            table.append([original_file, opensegspim_file, mse])
                            
with open('errors.csv','w') as file:
    for item in table:
        line = "%s \t %s \t %s \n" %(item[0], item[1], item[2])
        file.write(line)