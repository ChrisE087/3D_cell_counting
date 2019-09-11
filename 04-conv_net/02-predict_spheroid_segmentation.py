import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import os
import nrrd
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tools.cnn import CNN
from tools import datagen
from tools import image_processing as impro
from tools import datatools
import cc3d

def threshold_filter(data, threshold):
        data_thresh = np.copy(data)
        data_thresh[data_thresh > threshold] = 1
        data_thresh[data_thresh <= threshold] = 0
        return data_thresh.astype(np.uint8)

#%%############################################################################
# Specify the parameters
###############################################################################

# Specify the patch sizes and strides in each direction (ZYX)
patch_sizes = (32, 32, 32)
#strides = (32, 32, 32)
strides = (28, 28, 28)
#strides = (16, 16, 16)

# Specify the border around a patch in each dimension (ZYX), which is removed
cut_border = (2,2,2)#None#(8,8,8)

# Specify the padding which is used for the prediction of the patches
padding = 'VALID'

# Specify which model is used
model_import_path = os.path.join(os.getcwd(), 'model_export', 'FINAL', 'dataset_mix', 'segmentations', '2019-08-28_22-40-35_1')

# Specify the standardization mode
standardization_mode = 'per_sample'

# Specify the linear output scaling factor !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
linear_output_scaling_factor = 1#1e11#409600000000

# Specify if the results are saved
save_results = True

#%%############################################################################
# Read the data
###############################################################################
#category = '48h'
#spheroid_name = 'C2-untreated_2.nrrd'
#path_to_spheroid = os.path.join('..', '..', '..', 'Daten', category, 'untreated', spheroid_name)

category = 'Fibroblasten'
spheroid_name = '10_draq5_normalized.nrrd'
path_to_spheroid = os.path.join('..', '..', '..', 'Daten2', category, spheroid_name)

#%%############################################################################
# Initialize the CNN
###############################################################################
cnn = CNN(linear_output_scaling_factor=linear_output_scaling_factor, 
          standardization_mode=standardization_mode)
cnn.load_model_json(model_import_path, 'model_json', 'best_weights')

#%%############################################################################
# Predict the density-map
###############################################################################
spheroid_new, segmentation, segmentation_thresholded = cnn.predict_segmentation(path_to_spheroid=path_to_spheroid, patch_sizes=patch_sizes, 
                                                               strides=strides, border=cut_border, padding=padding, threshold=0.95, label=True)


# Testing
segmentation_thresholded = threshold_filter(segmentation, threshold=0.9)
segmentation_thresholded = cc3d.connected_components(segmentation_thresholded, connectivity=6)
print(np.max(segmentation_thresholded))

plt.figure()
plt.imshow(spheroid_new[int(spheroid_new.shape[0]/2),])
plt.figure()
plt.imshow(segmentation[int(segmentation.shape[0]/2),])
plt.figure()
plt.imshow(segmentation_thresholded[int(segmentation_thresholded.shape[0]/2),])
#segmentation_thresholded = np.transpose(segmentation_thresholded, axes=(2,1,0))
print(np.max(segmentation_thresholded))


#%%############################################################################
# Save the results
###############################################################################
if save_results == True:
    nrrd.write(category+'-'+spheroid_name+'.nrrd', spheroid_new)
    nrrd.write(category+'-'+spheroid_name+'-segmentation.nrrd', segmentation)
    nrrd.write(category+'-'+spheroid_name+'-segmentation_thresh.nrrd', segmentation_thresholded)
