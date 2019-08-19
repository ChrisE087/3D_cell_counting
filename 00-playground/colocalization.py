import nrrd
import numpy as np
import os
import matplotlib.pyplot as plt

def colocalize(segmentation, colocalization_channel, colocalization_threshold, make_colocalization_segmentation=False):
    
    num_of_colocalized_cells = 0
    
    # Check if the dimensions are the same
    if segmentation.shape != colocalization_channel.shape:
        print('Aborting. Segmentation and Colocalization channel have different shapes.')
        return
    
    # Check how many different labels there are in the segmentation -> get the labels
    labels = np.unique(segmentation)
    
    # Get the number of cells in the spheroid -> number of different labels
    num_of_cells = len(labels)
    
    # Make an array, where all labels are recorded, if the colocalization has 
    # found a signal in c1 on the segmentation with the label l. All other labels
    # are set to zero
    if make_colocalization_segmentation == True:
        colocalization_segmentation = np.copy(segmentation)
    else:
        colocalization_segmentation = None
    
    # Record the mean and label in a table
    result_table = []
    result_table.append(['Label', 'Colocalization', 'Mean', 'Standard-deviation', 'Variance'])
    
    # Iterate over all n labels
    for n in range(len(labels)):
        # Get the label
        l = labels[n]
        
        # Make a boolean mask of c2 -> All values that equal to the actual label 
        # in c2 are true (->mask). Then select all values in c1, where the mask is True
        values = colocalization_channel[segmentation == l]
        
        # Calculate the statistics over these values
        mean = np.mean(values)
        std = np.std(values)
        var = np.var(values)
        
        # Set the label to zero, if no colocalization was found
        if mean < colocalization_threshold:
            if make_colocalization_segmentation == True:
                colocalization_segmentation[segmentation == l] = 0
            result_table.append([l, 'False', mean, std, var])
        else:
            num_of_colocalized_cells = num_of_colocalized_cells + 1
            result_table.append([l, 'True', mean, std, var])       
    return result_table, num_of_cells, num_of_colocalized_cells, colocalization_segmentation
        
    
# Channel with proliferating cells
c1_file = os.path.join('..', '..', '..', 'Daten', '24h', 'untreated', 'C1-untreated_1.1.nrrd')
c1, c1_header = nrrd.read(c1_file)

# Channel with segmented nucleis
c2_file = os.path.join('..', '..', '..', 'Daten', '24h', 'untreated', 'C2-untreated_1.1_OpenSegSPIMResults_', 'Nucleisegmentedfill2r_labelled_filtered.nrrd')
c2, c2_header = nrrd.read(c2_file)

# Set the colocalization-threshold
colocalization_threshold = 10.0 #2*mean_of_background_noise

plt_z = 150

result_table, num_of_cells, num_of_colocalized_cells, colocalization_segmentation = colocalize(c2, c1, colocalization_threshold)



plt.imshow(c1[:,:,plt_z])
plt.imshow(c2[:,:,plt_z])
plt.imshow(colocalization_segmentation[:,:,plt_z])


nrrd.write('proliferating_nucleis.nrrd', data=colocalization_segmentation, header=c1_header)
