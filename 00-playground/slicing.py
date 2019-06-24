import sys
sys.path.append("..")
import numpy as np
import nrrd
import matplotlib.pyplot as plt
from tools import image_processing as impro


# Load the data
data, header = nrrd.read('test_data/24h_C1-untreated_1.2-0424.nrrd')
X = data[0,]
Y = data[1,]

# Slice the inner volume out of the data
border = 16
x_start = border
x_end = X.shape[0]-border
y_start = border
y_end = X.shape[1]-border
z_start = border
z_end = X.shape[2]-border
X_slice = X[x_start:x_end, y_start:y_end, z_start:z_end]
Y_slice = Y[x_start:x_end, y_start:y_end, z_start:z_end]

# Plot the results
plt_slice = 32
plt.imshow(X[:,:,plt_slice])
plt.imshow(Y[:,:,plt_slice])

plt_slice = 16
plt.imshow(X_slice[:,:,plt_slice])
plt.imshow(Y_slice[:,:,plt_slice])

# Test the method
X_slice = impro.get_inner_slice(data=X, border=(16,16,16))
Y_slice = impro.get_inner_slice(data=Y, border=(16,16,16))

# Plot the results
plt_slice = 32
plt.imshow(X[:,:,plt_slice])
plt.imshow(Y[:,:,plt_slice])

plt_slice = 16
plt.imshow(X_slice[:,:,plt_slice])
plt.imshow(Y_slice[:,:,plt_slice])
