import nrrd
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from tools import image_processing as impro


path_to_data = os.path.join('test_data', 'test_patch.nnrd')
data, header = nrrd.read(path_to_data)

X = data[0,]
y = data[1,]

plt.imshow(X[:,:,25])

###############################################################################
# Data normalization
###############################################################################
X_norm = tf.keras.utils.normalize(X, axis=-1, order=2)

# Check min and max
print(np.min(X_norm))
print(np.max(X_norm))

plt.imshow(X_norm[:,:,25])

###############################################################################
# Data standardization
###############################################################################

# Transpose the input-data from XYZ to YXZ for use with TensorFlow
X_std = np.transpose(X, axes=(1,0,2))

# Standadize the data with TensorFlow
sess = tf.Session()
with sess.as_default():
    X_std = tf.image.per_image_standardization(X_std).eval()

# Check the mean and standard deviation
print(np.mean(X_std))
print(np.std(X_std))

# Transpose back to XYZ
X_std = np.transpose(X_std, axes=(1,0,2))
plt.imshow(X_std[:,:,25])

# Test the function
X_std = impro.standardize_data_tf(X)

# Check the mean and standard deviation
print(np.mean(X_std))
print(np.std(X_std))

plt.imshow(X_std[:,:,25])