import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import os
import nrrd
import math
from tools import image_processing as impro
from tools import datatools

path_to_data = os.path.join('test_data', '24h_C1-untreated_1.2-0424.nrrd')
data, header = nrrd.read(path_to_data)
X = data[0,]
padding = np.zeros(shape=(64,64,64))

plt.imshow(X[50,])
print(X.mean())
print(X.std())

mean = X.mean()
std = X.std()
adj_std = max(std, 1.0/math.sqrt(X.size))
X_std = (X-mean)/adj_std

print(X_std.mean())
print(X_std.std())

mean = padding.mean()
std = padding.std()
adj_std = max(std, 1.0/math.sqrt(padding.size))
padding_std = (padding-mean)/adj_std

print(padding_std.mean())
print(padding_std.std())

path_to_data = os.path.join('..', '00-playground_NN', 'mini_dataset', 'inner_part')
files = os.listdir(path_to_data)

X = np.zeros(shape=(len(files), 32, 32, 32))
y = np.zeros_like(X)

X_std = np.zeros_like(X)
y_std = np.zeros_like(y)

for i in range(len(files)):
    data, header = nrrd.read(os.path.join(path_to_data, files[i]))
    X[i,] = data[0,]
    y[i,] = data[1,]
    
plt.imshow(X[50,20,])
print(np.mean(X[50,]))
print(np.std(X[50,]))

for i in range(X.shape[0]):
    mean = X[i,].mean()
    std = X[i,].std()
    adj_std = max(std, 1.0/math.sqrt(X[i].size))
    X_std[i,] = (X[i,] - mean)/adj_std 

plt.imshow(X_std[50,20,])
print(np.mean(X_std[50,]))
print(np.std(X_std[50,]))


X_std = impro.standardize_3d_images(X)
plt.imshow(X_std[100,20,])
print(np.mean(X_std[100,]))
print(np.std(X_std[100,]))

X_std = impro.standardize_3d_images(X[100,])
plt.imshow(X_std[20,])
print(np.mean(X_std))
print(np.std(X_std))

X_data, y_data = datatools.load_data2(path_to_dataset=path_to_data, data_list=files, input_shape=(32,32,32), 
              standardize=True, border=None)

plt.imshow(X_data[100,20,])
print(np.mean(X_data[100,]))
print(np.std(X_data[100,]))
