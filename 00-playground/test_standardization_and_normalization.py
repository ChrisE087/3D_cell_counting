import sys
sys.path.append("..")
import os
import numpy as np
import nrrd
from tools import image_processing as impro

path_to_data = os.path.join('test_data', 'Y.nrrd')
y, y_header = nrrd.read(path_to_data)

print('#### Standardizating data ####')

# Standardizize
print('Mean before standardization: ', np.mean(y))
print('Sigma before standardization: ', np.std(y))

y_standardizated, y_mean, y_sigma = impro.standardizate_data(y)

print('Mean after standardization: ', np.mean(y_standardizated))
print('Sigma after standardization: ', np.std(y_standardizated))

# Revert Standardization
y_reverted_standardization = impro.unstandardizate_data(y_standardizated, y_mean, y_sigma)

print('Mean after standardization reverted: ', np.mean(y_reverted_standardization))
print('Sigma after standardization reverted: ', np.std(y_reverted_standardization))

print('#### Normalizing data ####')

# Normalize
print('Min before normalization: ', np.min(y))
print('Max before normalization: ', np.max(y))

y_normalized, y_max, y_min = impro.normalize_data(y)

print('Min after normalization: ', np.min(y_normalized))
print('Max after normalization: ', np.max(y_normalized))

# Revert normalization
y_reverted_normalization = impro.unnormalize_data(y_normalized, y_max, y_min)

print('Min after normalization reverted: ', np.min(y_reverted_normalization))
print('Max after normalization reverted: ', np.max(y_reverted_normalization))