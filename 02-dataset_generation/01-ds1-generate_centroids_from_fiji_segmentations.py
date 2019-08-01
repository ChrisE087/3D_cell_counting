import sys
sys.path.append("..")
import os
import numpy as np
import nrrd
from tools import image_processing as impro

path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', '..', 'Daten'))

table = []
table.append(['Spheroid', 'Number of cells (cell-volumes > 0um^3)', 'Number of cells (ground-truth, cell-volumes > 3um^3)'])

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
                            #nrrd_file = os.path.join(data_dir, filename)
                            seg_file = os.path.join(res_dir, 'Nucleisegmentedfill2r.nrrd')
                            print('Processing file: ', seg_file)
                            seg_raw, seg_header = nrrd.read(seg_file) #XYZ
                            spacings = seg_header.get('space directions')
                            spacings = [spacings[0,0], spacings[1,1], spacings[2,2]]
                            spacings = np.array(spacings)
                            all_centroids, all_statistics = impro.get_centroids(seg_raw, spacings, 0.) # Unfiltered, only for documentation purposes
                            centroids, statistics = impro.get_centroids(seg_raw, spacings, 3.)
                            nrrd_centroids_file = os.path.join(res_dir, 'centroids_fiji_seg.nrrd')
                            nrrd.write(nrrd_centroids_file, data=centroids, header=seg_header, index_order='F')
                            # Log the number of cells in a table
                            spheroid_title = res_dir.split(os.path.sep)[2] + '->' + spheroid_name
                            table.append([spheroid_title, np.sum(all_centroids), np.sum(centroids)])
                            
with open('cell_numbers_dataset1_fiji_segmentations_filtered.txt','w') as file:
    for item in table:
        line = "%s \t %s \t %s\n" %(item[0], item[1], item[2])
        file.write(line)