import sys
sys.path.append("..")
import os
import nrrd
import numpy as np
import sys
from tools import image_processing as impro
from tools.cnn import CNN

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
    
def get_files_in_directory(a_dir):
    files = [f for f in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, f))]
    return files


#%%############################################################################
# Specify the parameters
###############################################################################

# Specify the data dir
path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', '..', 'Daten2'))

# Specify the suffix of the ground-truth
centroids_suffix = '-centroids' # '-centroids'

# Specify the patch sizes and strides in each direction (ZYX)
patch_sizes = (32, 32, 32)
strides = (32, 32, 32)

# Specify the border around a patch in each dimension (ZYX), which is removed
cut_border = None #(8,8,8)

# Specify the padding which is used for the prediction of the patches
padding = 'VALID'

# Specify which model is used
model_import_path = os.path.join('..', '04-conv_net', 'model_export', 'dataset_mix', '2019-08-13_22-25-12_1_3_train_samples_fiji_and_mathematica_segmentations')

# Specify the standardization mode
standardization_mode = 'per_sample'

# Specify the linear output scaling factor !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
linear_output_scaling_factor = 1#1e11#409600000000

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
table.append(['Category', 'Spheroid', 'Ground-Truth number of cells (cell-volumes > 3um^3)', 'Number of cells (predicted)', 'Absolute difference', 'Percentual difference'])

subdirs1 = get_immediate_subdirectories(path_to_data)
for subdir1 in subdirs1:
    print('-------------')
    # Spheroid type level
    spheroid_dir = os.path.join(path_to_data, subdir1)
    spheroid_files = get_files_in_directory(spheroid_dir)
    for spheroid_file in spheroid_files:
        if spheroid_file.endswith('.nrrd'):
            spheroid_name = os.path.splitext(spheroid_file)[0]
            spheroid_file = os.path.join(spheroid_dir, spheroid_file)
            subdirs2 = get_immediate_subdirectories(spheroid_dir)
            for subdir2 in subdirs2:
                res_dir = os.path.abspath(os.path.join(spheroid_dir, subdir2))
                centroid_files = get_files_in_directory(res_dir)
                for centroids_file in centroid_files:
                    if spheroid_name + centroids_suffix in centroids_file and centroids_file.endswith('.nrrd'):
                        spheroid_file = os.path.abspath(spheroid_file)
                        print('Predicting: ', spheroid_file)
                        
                        # Load the ground-truth
                        centroids_file = os.path.join(os.path.abspath(spheroid_dir), subdir2, centroids_file)
                        centroids, centroids_header = nrrd.read(centroids_file) # XYZ
                        num_of_cells_ground_truth = np.sum(centroids).astype(np.int)
                        
                        # Predict the segmentation
                        spheroid_new, segmentation, segmentation_thresholded = cnn.predict_segmentation(path_to_spheroid=spheroid_file, patch_sizes=patch_sizes, 
																									 strides=strides, border=cut_border, padding=padding, threshold=0.93, label=True)
                        
                        # Greatest label represents the number of cells in the spheroid
                        num_of_cells_predicted = np.max(segmentation_thresholded).astype(np.int)
                        
                        # Calculate the difference from the ground-truth
                        abs_diff = num_of_cells_predicted - num_of_cells_ground_truth
                        perc_diff = 100-(num_of_cells_predicted*100/num_of_cells_ground_truth)
                        
                        # Log the number of cells in a table
                        #spheroid_title = res_dir.split(os.path.sep)[2] + '->' + spheroid_name
                        category = res_dir.split(os.path.sep)[2]
                        table.append([category, spheroid_name, num_of_cells_ground_truth, num_of_cells_predicted, abs_diff, perc_diff])
                        
##%% Save the results in a table
with open('predicted_cell_numbers_dataset2_mathematica_segmentations.txt','w') as file:
    for item in table:
        line = "%s \t %s \t %s \t %s \t %s \t %s\n" %(item[0], item[1], item[2], item[3], item[4], item[5])
        file.write(line)