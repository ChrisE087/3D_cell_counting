import numpy as np
import tensorflow as tf
import nrrd
import os
import matplotlib.pyplot as plt


input_data_path = os.path.join('test_data', 'test_output_data.nrrd')

# Load the volume
input_data, input_data_header = nrrd.read(input_data_path) # XYZ

# Normalize the data -> Values between 0 and 1
scaled_data = (input_data - np.min(input_data))/(np.max(input_data)-np.min(input_data))
scaled_data = scaled_data.astype('float')

# Scale the data -> Values between 0 and 65535
scaled_data = scaled_data*65535
scaled_data = scaled_data.astype('uint16')

###############################################################################
# Generate patches
###############################################################################

# Undo the scaling
output_data = scaled_data.astype('float')
output_data = scaled_data/65535

# Undo the normalization
output_data = output_data.astype('float64')
output_data = output_data*(np.max(input_data)-np.min(input_data))+np.min(input_data)

# Compare the sum
np.sum(input_data)
np.sum(output_data)
