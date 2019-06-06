import numpy as np
import tensorflow as tf
import nrrd
import os
import matplotlib.pyplot as plt

def normalize_data(input_data):
    max_val = np.max(input_data)
    min_val = np.min(input_data)
    normalized_data = (input_data - min_val)/(max_val-min_val)
    normalized_data = normalized_data.astype('float')
    return normalized_data, max_val, min_val

def undo_data_normalization(normalized_data, max_val, min_val):
    normalized_data = normalized_data.astype('float64')
    unnormalized_data = normalized_data*(max_val-min_val)+min_val
    return unnormalized_data

def scale_data(data, factor):
    scaled_data = data*factor
    scaled_data = scaled_data.astype('uint16')
    return scaled_data

def unscale_data(data, factor):
    unscaled_data = data.astype('float')
    unscaled_data = data/factor
    return unscaled_data
    


input_data_path = os.path.join('test_data', 'test_output_data.nrrd')

# Load the volume
input_data, input_data_header = nrrd.read(input_data_path) # XYZ

# Normalize the data -> Values between 0 and 1
normalized_data, max_val, min_val = normalize_data(input_data)

# Scale the data -> Values between 0 and 65535
scaled_data = scale_data(normalized_data, 65535)

###############################################################################
# Generate patches
###############################################################################

# Undo the scaling
unscaled_data = unscale_data(scaled_data, 65535)

# Undo the normalization
output_data = undo_data_normalization(unscaled_data, max_val, min_val)

# Compare the sum
print('Sum over input data: ', np.sum(input_data))
print('Sum over output data: ', np.sum(output_data))

