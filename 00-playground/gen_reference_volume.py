import sys
sys.path.append("..")
import numpy as np
import nrrd
import matplotlib.pyplot as plt
import os

# Load data
file_path = os.path.join('..', '..', '..', 'Daten', 'extracted_cells')
files = os.listdir(file_path)
cells = []
for i in range(len(files)):
    data, header = nrrd.read(os.path.join(file_path, files[i])) #XYZ
    data = np.transpose(data, axes=(2,1,0)) #ZYX
    cells.append(data)
    
# Generate a 250x250x250 volume with noise and padding
vol = np.zeros(shape=(250,250,250))
#vol = np.random.randint(30, size=(250, 250, 250))
plt.imshow(vol[50,])

for i in range(len(cells)):
    z_begin = np.random.randint(low=50, high=200)
    y_begin = np.random.randint(low=50, high=200)
    x_begin = np.random.randint(low=50, high=200)
    cell = cells[i]
    z_end = z_begin + cell.shape[0]
    y_end = y_begin + cell.shape[1]
    x_end = x_begin + cell.shape[2]
    vol[z_begin:z_end, y_begin:y_end, x_begin:x_end] += cell
    
plt.imshow(vol[100,])
vol = np.transpose(vol, axes=(2,1,0)) #XYZ
nrrd.write('test_volume', vol)
