import sys
sys.path.append("..")
import os
import numpy as np
import nrrd
from tools import image_processing as impro
from tools.cnn import CNN
import tensorflow as tf

#%%############################################################################
# Specify the parameters
###############################################################################

# Specify the data dir
path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', '..', 'Daten'))

# Spceify the results dir
path_to_colocalization_results = 'results'

# Specify the filename of the ground-truth
centroids_filename = 'centroids_fiji_seg.nrrd' # 'centroids_fiji_seg.nrrd' or 'centroids_opensegspim_seg.nrrd'

# Specify the patch sizes and strides in each direction (ZYX)
patch_sizes = (32, 32, 32)
strides = (16, 16, 16)

# Specify the border around a patch in each dimension (ZYX), which is removed
cut_border = (8,8,8)

# Specify the padding which is used for the prediction of the patches
padding = 'VALID'

# Specify which model is used
model_import_path = os.path.join('..', '04-conv_net', 'model_export', 'dataset1', '2019-08-10_09-13-46_1_3_train_samples_fiji_SEGMENTATIONS_crossentropy_256_epochs')

# Specify the standardization mode
standardization_mode = 'per_sample'

# Specify the linear output scaling factor !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
linear_output_scaling_factor = 1#1e11#409600000000

colocalization_threshold = 10.0

#%%############################################################################
# Initialize the CNN
###############################################################################
cnn = CNN(linear_output_scaling_factor=linear_output_scaling_factor, 
          standardization_mode=standardization_mode)
cnn.load_model_json(model_import_path, 'model_json', 'best_weights')

table = []
table.append(['Cultivation-period', 'Spheroid', 'Ground-Truth number of cells (cell-volumes > 3um^3)', 'Number of cells (predicted)', 'Absolute difference', 'Percentual difference', 'Number of colocalized cells', 'Percentual number of colocalized cells'])

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
                        if spheroid_name in subdir and 'C2' in subdir:
                            # Get the filepaths
                            spheroid_file = os.path.join(data_dir, filename)
                            colocalization_filename = 'C1-'+filename[3:]
                            colocalization_file = os.path.join(data_dir, colocalization_filename)
                            centroids_file = os.path.join(res_dir, 'centroids_fiji_seg.nrrd')
                            
                            # Get the cultivation-period and make the export-path
                            cultivation_period = centroids_file.split(os.path.sep)[2]
                            export_path = os.path.join(path_to_colocalization_results, cultivation_period)
                            if not os.path.exists(export_path):
                                os.makedirs(export_path)
                            
                            print('################################################################################################')
                            print('Colocalization file: ', colocalization_file)
                            
                            # Load the ground-truth
                            centroids_file = os.path.join(res_dir, centroids_filename)
                            print('Loading the ground truth number of cells: ', centroids_file)
                            centroids, centroids_header = nrrd.read(centroids_file) #XYZ
                            num_of_cells_ground_truth = np.sum(centroids).astype(np.int)
                            print('Number of cells in spheroid (ground-truth): ', num_of_cells_ground_truth)
                            
                            # Load the colocalization file
                            print('Loading the colocalization file: ', colocalization_file)
                            colocalization, colocalization_header = nrrd.read(colocalization_file)
                            
                            # Generate image patches from the colocalization file and restore the patches to the volume, so it has the same dimension as the predicted segmentation
                            print('Patching and restoring the colocalization file: ', colocalization_file)
                            session = tf.Session()
                            patches_colocalization = impro.gen_patches(session=session, data=colocalization, patch_slices=patch_sizes[0], patch_rows=patch_sizes[1], 
                                                        patch_cols=patch_sizes[2], stride_slices=strides[0], stride_rows=strides[1], 
                                                        stride_cols=strides[2], input_dim_order='XYZ', padding=padding)
                            colocalization_restored = impro.restore_volume(patches=patches_colocalization, border=cut_border, output_dim_order='XYZ')
                            
                            # Predict the segmentation of the spheroid
                            print('Predicting the segmentation for: ', spheroid_file)
                            spheroid_new, segmentation, segmentation_thresholded = cnn.predict_segmentation(path_to_spheroid=spheroid_file, patch_sizes=patch_sizes, 
																									 strides=strides, border=cut_border, padding=padding, threshold=0.95, label=True, session=session)
                            
                            session.close()
                            tf.reset_default_graph()
                            
                            # Greatest label represents the number of cells in the spheroid
                            num_of_cells_predicted = np.max(segmentation_thresholded).astype(np.int)
                            print('Number of cells in spheroid (predicted): ', num_of_cells_predicted)
                            
                            # Calculate the difference from the ground-truth
                            abs_diff = num_of_cells_ground_truth - num_of_cells_predicted
                            perc_diff = 100-(num_of_cells_predicted*100/num_of_cells_ground_truth)
                            
                            # Colocalize the cells in the colocalization channel that correspond to the label l
                            print('Calculating the colocalization between ', spheroid_file, ' and ', colocalization_file)
                            result_table, num_of_cells, num_of_colocalized_cells, colocalization_segmentation = impro.colocalize(segmentation=segmentation_thresholded, colocalization_volume=colocalization_restored, colocalization_threshold=colocalization_threshold, make_colocalization_segmentation=True)
                            
                            perc_num_of_colocalized_cells = num_of_colocalized_cells*100/num_of_cells_predicted
                            
                            # Export the results
                            export_suffix = colocalization_filename[2:-5]
                             
                            print('Exporting the spheroid')
                            spheroid_export_path = os.path.join(path_to_colocalization_results, cultivation_period, 'C2'+export_suffix+'.nrrd')
                            nrrd.write(spheroid_export_path, spheroid_new.astype(np.uint8), colocalization_header)
                            
                            print('Exporting the colocalization channel')
                            colocalization_channel_export_path = os.path.join(path_to_colocalization_results,  cultivation_period, 'C1'+export_suffix+'.nrrd')
                            nrrd.write(colocalization_channel_export_path, colocalization_restored.astype(np.uint8), colocalization_header)
                            
                            print('Exporting the segmentation')
                            segmentation_export_path = os.path.join(path_to_colocalization_results,  cultivation_period, 'C2'+export_suffix+'_seg.nrrd')
                            nrrd.write(segmentation_export_path, segmentation_thresholded.astype(np.uint16), colocalization_header)
                            
                            print('Exporting the colocalized segmentation')
                            colocalized_segmentation_export_path = os.path.join(path_to_colocalization_results,  cultivation_period, 'C2'+export_suffix+'_seg_colocalized.nrrd')
                            nrrd.write(colocalized_segmentation_export_path, colocalization_segmentation.astype(np.uint16), colocalization_header)
                            
                            print('Exporting the colocalization statistics')
                            colocalization_table_export_path = os.path.join(path_to_colocalization_results,  cultivation_period, export_suffix+'_colocalization_stats.txt')
                            with open(colocalization_table_export_path,'w') as file:
                                for item in result_table:
                                    line = "%s \t %s \t %s \t %s \t %s\n" %(item[0], item[1], item[2], item[3], item[4])
                                    file.write(line)
                            
                            # Log the number of cells in a table
                            #spheroid_title = res_dir.split(os.path.sep)[2] + '->' + spheroid_name
                            cultivation_period = res_dir.split(os.path.sep)[2]
                            table.append([cultivation_period, spheroid_name, num_of_cells_ground_truth, num_of_cells_predicted, abs_diff, perc_diff, num_of_colocalized_cells, perc_num_of_colocalized_cells])

with open(os.path.join(path_to_colocalization_results, 'cell_numbers_dataset1_fiji_segmentations_filtered.txt'),'w') as file:
    for item in table:
        line = "%s \t %s \t %s \t %s \t %s \t %s \t %s \t %s\n" %(item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7])
        file.write(line)