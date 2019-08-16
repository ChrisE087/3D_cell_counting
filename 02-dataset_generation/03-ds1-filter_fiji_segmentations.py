import sys
sys.path.append("..")
import os
import numpy as np
import nrrd
from tools import image_processing as impro

path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', '..', 'Daten'))

table = []
table.append(['Spheroid', 'Number of cells befor filtering (max label)', 'Number of cells befor filtering (max label)'])

# Specify the excluded volume size of the segmentations that are set to zero
excluded_volume_size = 3 #um^3

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
                            seg_file = os.path.join(res_dir, 'Nucleisegmentedfill2r_labelled.nrrd')
                            print('Processing file: ', seg_file)
                            segmentation, header = nrrd.read(seg_file) #XYZ
                            space_directions = header.get('space directions')
                            spacings = np.array((space_directions[0,0],space_directions[1,1], space_directions[2,2]))
                            segmentation_filtered = impro.filter_segmentation(segmentation, spacings, excluded_volume_size)
                            filtered_segmentation_file = os.path.join(res_dir, 'Nucleisegmentedfill2r_labelled_filtered.nrrd')
                            nrrd.write(filtered_segmentation_file, data=segmentation_filtered, header=header, index_order='F')
                            # Log the number of cells, min and max in a table
                            spheroid_title = res_dir.split(os.path.sep)[2] + '->' + spheroid_name
                            table.append([spheroid_title, np.max(segmentation), np.max(segmentation_filtered)])
                            
with open('cell_numbers_dataset1_fiji_segmentations_filtered.txt','w') as file:
    for item in table:
        line = "%s \t %s \t %s\n" %(item[0], item[1], item[2])
        file.write(line)