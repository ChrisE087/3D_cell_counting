from __future__ import with_statement
import sys
sys.path.append("..")
import os
import numpy as np
import nrrd
from tools import image_processing as impro
from tools.cnn import CNN

#%%############################################################################
# Specify the parameters
###############################################################################

# Specify the data dir
path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', '..', 'Daten'))

# Specify the patch sizes and strides in each direction (ZYX)
patch_sizes = (32, 32, 32)
strides = (32, 32, 32)

# Specify the border around a patch in each dimension (ZYX), which is removed
cut_border = None #(8,8,8)

# Specify the padding which is used for the prediction of the patches
padding = 'VALID'

# Specify which model is used
model_import_path = os.path.join('..', '04-conv_net', 'model_export', 'BEST', '2019-08-02_14-08-13_100000.0')

# Specify the standardization mode
standardization_mode = 'per_sample'

# Specify the linear output scaling factor !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
linear_output_scaling_factor = 1e5#1e11#409600000000

# Specify if the results are saved
save_results = False

#%%############################################################################
# Initialize the CNN
###############################################################################
cnn = CNN(linear_output_scaling_factor=linear_output_scaling_factor, 
          standardization_mode=standardization_mode)
cnn.load_model_json(model_import_path, 'model_json', 'best_weights')


#%%############################################################################
# Predict the density-map
###############################################################################

table = []
table.append(['Spheroid', 'Ground-Truth number of cells (cell-volumes > 3um^3)', 'Number of cells (predicted)', 'Absolute difference', 'Percentual difference'])

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
                            spheroid_file = os.path.join(data_dir, filename)
                            print('Predicting: ', spheroid_file)
                            
                            # Load the ground-truth
                            centroids_file = os.path.join(res_dir, 'centroids_fiji_seg.nrrd')
                            centroids, centroids_header = nrrd.read(centroids_file) #XYZ
                            num_of_cells_ground_truth = np.sum(centroids)

                            # Predict the density-map
                            spheroid_new, density_map, num_of_cells_predicted = cnn.predict_spheroid(path_to_spheroid=spheroid_file, patch_sizes=patch_sizes, 
																									 strides=strides, border=cut_border, padding=padding)
                            
                            # Calculate the difference from the ground-truth
                            abs_diff = num_of_cells_ground_truth - num_of_cells_predicted
                            perc_diff = 100-(num_of_cells_predicted*100/num_of_cells_ground_truth)
                            
                            # Log the number of cells in a table
                            spheroid_title = res_dir.split(os.path.sep)[2] + '->' + spheroid_name
                            table.append([spheroid_title, num_of_cells_ground_truth, num_of_cells_predicted, abs_diff, perc_diff])
                            
with open('predicted_cell_numbers_dataset1_fiji_segmentations.txt','w') as file:
    for item in table:
        line = "%s \t %s \t %s \t %s \t %s\n" %(item[0], item[1], item[2], item[3], item[4])
        file.write(line)