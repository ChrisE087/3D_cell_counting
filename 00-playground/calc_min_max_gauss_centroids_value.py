import sys
sys.path.append("..")
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import nrrd
from tools import image_io as bfio
from tools import image_processing as impro

# Specify the size of each patch
size_z = 32
size_y = 32
size_x = 32

path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Daten'))
path_to_dataset = os.path.join('dataset')

table = [['Name', 'Minimum value', 'Maximum value']]

for directory in os.listdir(path_to_data):
    data_dir = os.path.join(path_to_data, directory, 'untreated')
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.nrrd'):
                name = os.path.splitext(filename)[0]
                for subdir in os.listdir(data_dir):
                    res_dir = os.path.join(data_dir, subdir)
                    if(os.path.isdir(res_dir)):
                        centroids_file = os.path.join(res_dir, 'gauss_centroids.nrrd')
                        print(centroids_file)
#                        Y, Y_header = nrrd.read(centroids_file) #XYZ
#                        centroid = centroids_file.split(os.path.sep)[2] + '->' + name
#                        min_val = np.min(Y)
#                        max_val = np.max(Y)
#                        table.append([centroid, min_val, max_val])
#                        
#with open('min_max.txt','w') as file:
#    for item in table:
#        line = "%s \t %s \t %s\n" %(item[0], item[1], item[2])
#        file.write(line)
#                            