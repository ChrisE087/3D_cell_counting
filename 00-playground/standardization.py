import sys
sys.path.append("..")
import nrrd
import os
import numpy as np

def standardize_data(data, mode='per_sample', epsilon=1e-8):
    """
    Standardizates the dataset in data, so it has a mean of zero and a
    unit-variance.
    
    Parameters:
    input_dataset (Numpy Array): (N)xSxHxW Numpy Array to standardize, where N
    is the number of images in the dataset, H and W is the height and width and
    S is the number of slices. If the size of the shape of data is equal to 4, 
    axis 0 is interpreted as the batch dimension. When the shape of data is 
    equal to 3, a batch dimension is added and later removed.
    mode (String): Mode can be 'per_slice' so slice-wise standardization is 
    performed, 'per_sample' where every 3D sample in the batch is standardized
    or 'per_batch' where the mean and standard-deviation of the 
    whole batch is calculated, so that the complete dataset has zero mean and
    unit-variance. If batch_dim=False and mode='per_batch', the 'per_batch'
    normalization is equal to 'per_sample' normalization.
    epsilon (float): Small number to avoid division of zero on uniform 
    datasets.
    
    Returns:
    standardized_dataset (Numpy Array): Standardizated dataset.
    """
    if np.size(data.shape) > 4:
        print('WARNING! Data is not standardized, because arrays of a \
              dimension > 4 are not supported.')
    
    remove_batch_dim = False
    # Add the batch dimension
    if np.size(data.shape) == 3:
        data = data[np.newaxis, ]
        remove_batch_dim = True
    
    # Standardize the data dependig on the specified mode
    if mode == 'per_slice':
        mean = data.mean(axis=(2,3), keepdims=True)
        std = data.std(axis=(2,3), keepdims=True)
        standardized_data = (data-mean)/(std+epsilon)
    elif mode == 'per_sample':
        mean = data.mean(axis=(1,2,3), keepdims=True)
        std = data.std(axis=(1,2,3), keepdims=True)
        standardized_data = (data-mean)/(std+epsilon)
    elif mode == 'per_batch':
        mean = data.mean()
        std = data.std()
        standardized_data = (data-mean)/(std+epsilon)
    else:
        print('WARNING! Data is not standardized, because the selected mode is\
              not supported. Modes are only "per_slice" for slice-wise\
              normalization, "per_sample" for sample-wise normalization or\
              "per_batch" for normalization of the whole batch.')
        return data
            
    # Remove the batch dimension
    if remove_batch_dim == True:
        standardized_data = standardized_data[0,]
        
    return standardized_data

#data_dir = os.path.join('..', '..', '..', 'Daten', 'extracted_cells')
#
#data = []
#
#for file_name in os.listdir(data_dir):
#    cell = nrrd.read(os.path.join(data_dir, file_name))
#    data.append(cell)
#    
#cell = data[0]
#cell = cell[0]

###############################################################################
# Load a batch of test data
###############################################################################

data_dir = os.path.join('..', '..', '..', 'Daten', 'dataset')
files = os.listdir(data_dir)

batch = np.zeros((10, 32, 32, 32))
for i in range(10):
    sample, header = nrrd.read(os.path.join(data_dir, files[i]))
    batch[i] = sample[0]

###############################################################################
# Standardize every slice in the batch
###############################################################################
epsilon = 1e-8
mean = batch.mean(axis=(2,3), keepdims=True)
std = batch.std(axis=(2,3), keepdims=True)
standardized_slices = (batch-mean)/(std+epsilon)

for i in range(standardized_slices.shape[0]):
    for j in range(standardized_slices.shape[1]):
        print('Mean: ', np.mean(standardized_slices[i,j,]))
        print('Standard deviation: ', np.std(standardized_slices[i,j,]))
        
###############################################################################
# Standardize every volume in the batch
###############################################################################
epsilon = 1e-8
mean = batch.mean(axis=(1,2,3), keepdims=True)
std = batch.std(axis=(1,2,3), keepdims=True)
standardized_volumes = (batch-mean)/(std+epsilon)

for i in range(standardized_volumes.shape[0]):
    print('Mean: ', np.mean(standardized_volumes[i,]))
    print('Standard deviation: ', np.std(standardized_volumes[i,]))
    
###############################################################################
# Standardize the whole batch
###############################################################################
epsilon = 1e-8
mean = batch.mean()
std = batch.std()
standardized_batch = (batch-mean)/(std+epsilon)

print('Mean: ', standardized_batch.mean())
print('Standard deviation: ', standardized_batch.std())

###############################################################################
# Test the method with different modes
###############################################################################
standardized_slices = standardize_data(data=batch, mode='per_slice', 
                                       epsilon=1e-8)
for i in range(standardized_slices.shape[0]):
    for j in range(standardized_slices.shape[1]):
        print('Mean: ', np.mean(standardized_slices[i,j,]))
        print('Standard deviation: ', np.std(standardized_slices[i,j,]))
        
standardized_volumes = standardize_data(data=batch, mode='per_sample', 
                                        epsilon=1e-8)
for i in range(standardized_volumes.shape[0]):
    print('Mean: ', np.mean(standardized_volumes[i,]))
    print('Standard deviation: ', np.std(standardized_volumes[i,]))
    
standardized_batch = standardize_data(data=batch, mode='per_batch', 
                                      epsilon=1e-8)
print('Mean: ', standardized_batch.mean())
print('Standard deviation: ', standardized_batch.std())

single_standardized_slices = standardize_data(data=batch[0,], mode='per_slice', 
                                      epsilon=1e-8)
for i in range(single_standardized_slices.shape[0]):
    print('Mean: ', np.mean(single_standardized_slices[i]))
    print('Standard deviation: ', np.std(single_standardized_slices[i]))
    
single_standardized_volume = standardize_data(data=batch[0,], mode='per_sample', 
                                       epsilon=1e-8)
print('Mean: ', np.mean(single_standardized_volume[:,]))
print('Standard deviation: ', np.std(single_standardized_volume[:,]))

single_standardized_volume = standardize_data(data=batch[0,], mode='per_batch',
                                              epsilon=1e-8)
print('Mean: ', np.mean(single_standardized_volume[:,]))
print('Standard deviation: ', np.std(single_standardized_volume[:,]))

