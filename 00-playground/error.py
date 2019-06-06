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

# Read the data
original_data, original_header = nrrd.read('test_data/C1-untreated_1.nrrd')
processed_data, processed_data_header = nrrd.read('test_data/C1-untreated_1_OriginalStack.nrrd')

# Convert to 8-bit
processed_data = (processed_data/256).astype('uint8')

# Convert to float
original_data = original_data.astype('float')
processed_data = processed_data.astype('float')

# Calculate the error
error = np.abs((original_data - processed_data))

# Calculate the mean error
mean_error = np.sum(error)/np.size(error)

# Plot the error
plt.imshow(original_data[:,:,100])
plt.imshow(processed_data[:,:,100])
plt.imshow(error[:,:,100])