import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import os
import nrrd

path_to_dataset = os.path.join('..', '..', '..', 'Daten', 'dataset')
files = os.listdir(path_to_dataset)
max_intensity = []
mean = []
cell_number = []

for i in range(len(files)):
    data, header = nrrd.read(os.path.join(path_to_dataset, files[i]))
    max_intensity.append(np.max(data[0,]))
    #mean.append(np.mean(data[0,])) # Mean is computationally intensive
    cell_number.append(np.sum(data[1,]))

fig = plt.figure()    
plt.scatter(max_intensity, cell_number, s=1)
plt.xlabel('Max Intensities')
plt.ylabel('Cell number')
fig.savefig('scatter.png', dpi=900)

fig = plt.figure()
plt.title('Distribution of maximum intensities (0-255)')
plt.hist(max_intensity, range=(0,255), bins=50)

fig = plt.figure()
plt.title('Distribution of maximum intensities (0-50)')
plt.hist(max_intensity, range=(0,50), bins=50)

fig = plt.figure()
plt.title('Distribution of cell numbers (0-25 cells)')
plt.hist(cell_number, range=(0,25), bins=50)

fig = plt.figure()
plt.title('Distribution of cell numbers (0-1 cells)')
plt.hist(cell_number, range=(0,1), bins=50)


# Collect files with little cells
cell_thresh = 0.1
plt_slice = 15
little_cells = []
for i in range(len(files)):
    if cell_number[i] <= 0.1:
        little_cells.append([files[i], cell_number[i]])

# Plot a random bunch of random patches with little cells
rows = cols = 8
fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row', figsize=(50, 50))
for i in range(rows):
    for j in range(cols):
        if j%2 == 0:
            sample_num = np.random.randint(0, len(little_cells))
            file = os.path.join(path_to_dataset, little_cells[sample_num][0])
            data, header = nrrd.read(file)
            
            ax[i, j].imshow(data[0,plt_slice,], interpolation='nearest')
            ax[i, j].title.set_text('Input max intensity value = ' + str(np.max(data[0,])))
            ax[i, j+1].imshow(data[1,plt_slice,], interpolation='nearest')
            ax[i, j+1].title.set_text('Target number of cells = ' + str(np.sum(data[1,])))
plt.show()

# Collect files with mostly noise (detection with max-intensity <= 50)
noise = []
for i in range(len(files)):
    if max_intensity[i] <= 70:
        noise.append([files[i], max_intensity[i]])
        
# Plot a random bunch of random patches with noisy data
rows = cols = 8
fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row', figsize=(50, 50))
for i in range(rows):
    for j in range(cols):
        if j%2 == 0:
            sample_num = np.random.randint(0, len(noise))
            file = os.path.join(path_to_dataset, noise[sample_num][0])
            data, header = nrrd.read(file)
            
            ax[i, j].imshow(data[0,plt_slice,], interpolation='nearest')
            ax[i, j].title.set_text('Input max intensity value = ' + str(np.max(data[0,])))
            ax[i, j+1].imshow(data[1,plt_slice,], interpolation='nearest')
            ax[i, j+1].title.set_text('Target number of cells = ' + str(np.sum(data[1,])))
plt.show()

# Collect files with mostly noise (detection with mean-intensity <= 50)
#noise2 = []
#for i in range(len(files)):
#    if mean[i] <= 50:
#        noise2.append([files[i], max_intensity[i]])
#
## Plot a random bunch of random patches with noisy data
#rows = cols = 8
#fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row', figsize=(50, 50))
#for i in range(rows):
#    for j in range(cols):
#        if j%2 == 0:
#            sample_num = np.random.randint(0, len(noise2))
#            file = os.path.join(path_to_dataset, noise2[sample_num][0])
#            data, header = nrrd.read(file)
#            
#            ax[i, j].imshow(data[0,plt_slice,], interpolation='nearest')
#            ax[i, j].title.set_text('Input max intensity value = ' + str(np.max(data[0,])))
#            ax[i, j+1].imshow(data[1,plt_slice,], interpolation='nearest')
#            ax[i, j+1].title.set_text('Target number of cells = ' + str(np.sum(data[1,])))
#plt.show()


