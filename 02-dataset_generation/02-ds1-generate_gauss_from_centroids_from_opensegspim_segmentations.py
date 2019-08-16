import sys
sys.path.append("..")
import os
import numpy as np
import nrrd
from tools import image_processing as impro

path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', '..', 'Daten'))

table = []
table.append(['Spheroid', 'Number of cells (sum over density-map)', 'Min value', 'Max value'])

for directory in os.listdir(path_to_data):
    data_dir = os.path.join(path_to_data, directory, 'untreated')
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.nrrd'):
                spheroid_name = os.path.splitext(filename)[0]
                #print('Actual file: ', spheroid_name)
                for subdir in os.listdir(data_dir):
                    res_dir = os.path.join(data_dir, subdir)
                    if(os.path.isdir(res_dir)):
                        if spheroid_name in subdir:
                            nrrd_file = os.path.join(data_dir, filename)
                            centroids_file = os.path.join(res_dir, 'centroids_opensegspim_seg.nrrd')
                            print('Processing file: ', centroids_file)
                            centroids, header = nrrd.read(centroids_file) #XYZ
                            gauss_centroids = impro.convolve_with_gauss(centroids, 50, 6)
                            nrrd_gauss_centroids_file = os.path.join(res_dir, 'gauss_centroids_opensegspim_seg.nrrd')
                            nrrd.write(nrrd_gauss_centroids_file, data=gauss_centroids, header=header, index_order='F')
                            # Log the number of cells, min and max in a table
                            spheroid_title = res_dir.split(os.path.sep)[2] + '->' + spheroid_name
                            table.append([spheroid_title, np.sum(gauss_centroids), np.min(gauss_centroids), np.max(gauss_centroids)])
                            
with open('gauss_cell_numbers_dataset1_opensegspim_segmentations_filtered.txt','w') as file:
    for item in table:
        line = "%s \t %s \t %s \t %s\n" %(item[0], item[1], item[2], item[3])
        file.write(line)