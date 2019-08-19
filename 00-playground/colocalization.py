import nrrd
import numpy as np
import os
import SimpleITK as sitk
import sys
import matplotlib.pyplot as plt
import cc3d
import copy

# Channel with proliferating cells
c1_file = os.path.join('..', '..', '..', 'Daten', '24h', 'untreated', 'C1-untreated_1.1.nrrd')
c1, c1_header = nrrd.read(c1_file)
#c1_file = os.path.join('test_data', 'colocalization_C1.nrrd')
#c1, c1_header = nrrd.read(c1_file)

# Channel with segmented nucleis
c2_file = os.path.join('..', '..', '..', 'Daten', '24h', 'untreated', 'C2-untreated_1.1_OpenSegSPIMResults_', 'Nucleisegmentedfill2r_labelled_filtered.nrrd')
c2, c2_header = nrrd.read(c2_file)
#c2_file = os.path.join('test_data', 'colocalization_C2_seg.nrrd')
#c2, c2_header = nrrd.read(c2_file)

# Set the colocalization-threshold
colocalization_threshold = 10.0 #2*mean_of_background_noise

# Check if the dimensions are the same
if c1.shape != c2.shape:
    print('Aborting. Channel 1 and channel 2 have different shapes.')
    sys.exit(-1)
    
plt_z = 65
plt.imshow(c1[:,:,plt_z])
plt.imshow(c2[:,:,plt_z])

# Check how many different labels there are in the segmentation -> get the labels
labels = np.unique(c2)

# Make an array, where all labels are recorded, if the colocalization has 
# found a signal in c1 on the segmentation with the label l. All other labels
# are set to zero
colocalized_cells = np.copy(c2)

# Recprd the mean and label in a table
table = []
table.append(['Label', 'Colocalization', 'Mean', 'Standard-deviation', 'Variance'])

# Iterate over all n labels
for n in range(len(labels)):
    # Get the label
    l = labels[n]
    
    # Make a boolean mask of c2 -> All values that equal to the actual label 
    # in c2 are true (->mask). Then select all values in c1, where the mask is True
    values = c1[c2 == l]
    
    # Calculate the mean over these values
    mean = np.mean(values)
    std = np.std(values)
    var = np.var(values)
    
    # Set the label to zero, if no colocalization was found
    if mean < colocalization_threshold:
        colocalized_cells[c2 == l] = 0
        table.append([l, 'False', mean, std, var])
    else:
        table.append([l, 'True', mean, std, var])

plt.imshow(c1[:,:,plt_z])
plt.imshow(c2[:,:,plt_z])
plt.imshow(colocalized_cells[:,:,plt_z])


nrrd.write('proliferating_nucleis.nrrd', data=colocalized_cells, header=c1_header)
