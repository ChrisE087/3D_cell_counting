import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import keras
import os
import time
import datetime
import nrrd
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tools.cnn import CNN
from tools import datagen
from tools import image_processing as impro
from tools import datatools

# Read the data
path_to_nuclei = os.path.join('..', '..', '..', 'Daten', '24h', 'untreated', 'C1-untreated_1.1_OpenSegSPIMResults_', 'gauss_centroids.nrrd')
#path_to_nuclei = os.path.join('24h_C1-untreated_1.1-00000197.nrrd')
data, header = nrrd.read(path_to_nuclei)

# Visualize min-max
print(np.min(data))
print(np.max(data))
plt.hist(data.flatten(), range=(0,0.0003), bins=100)

# Scale and visualize min-max
data_scaled = np.copy(data)
data_scaled = data_scaled*1e20
print(np.min(data_scaled))
print(np.max(data_scaled))
plt.hist(data_scaled.flatten(), range=(0,5.6307764e+16), bins=255)


plt_slice = 25
plt.imshow(data[0,plt_slice,])
plt.imshow(data[1,plt_slice,])
print(np.sum(data[1,]))
print(np.sum(data[0,]))
print(np.max(data[0,]))
a = data[0,]
