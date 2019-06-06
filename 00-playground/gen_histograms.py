import sys
sys.path.append("..")
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import nrrd
import SimpleITK as sitk

# Read the data into pandas data frame
path_to_data = os.path.join('..', '..', 'Daten', '24h', 'untreated', 'C1-untreated_1.1_OpenSegSPIMResults_')
data_file = os.path.join(path_to_data, 'Nuclei_measurement_results.txt')
data = pd.read_csv(data_file, sep='\t', header=0, usecols=['Volume (um^3)']) 

# Convert the volume column to numpy array
volumes_segspim = data.values


# Read the nuclei segmentation and calculate the volumes of each nuclei
seg_file = os.path.join(path_to_data, 'Nucleisegmentedfill.nrrd')
raw_data, header = nrrd.read(seg_file) #XYZ

# Transpose the numpy array from XYZC to CZYX for the use with SimpleITK
raw_data = np.transpose(raw_data, axes=[2,1,0]) # ZYX

# Make a SimpleITK out of the numpy array
image = sitk.GetImageFromArray(raw_data, isVector=False) # XYZ
image.SetSpacing(header.get('spacings'))

# Get The Connected Components of the volume image. All intensities greater 
# than 0 are taken into account for the labeling
cc = sitk.ConnectedComponent(image>0)

# Calculate the statitics of the labeled regions
stats = sitk.LabelIntensityStatisticsImageFilter()
stats.Execute(cc, image)

volumes_sitk = []
# Print the statistics
for l in stats.GetLabels():
    volumes_sitk.append(stats.GetPhysicalSize(l))
    #print("Label: {0} Size/Volume: {1}".format(l, stats.GetPhysicalSize(l)))
    #print('_____________________________________________________')
    
volumes_sitk = np.array(volumes_sitk, dtype='float64')
volumes_sitk = np.sort(volumes_sitk)

# Create a histogram out of the data
figname = path_to_data.split('\\')
figname = "%s_%s.png" % (figname[3], figname[5])
bins = np.arange(0, 500, 10)
fig = plt.figure()
plt.suptitle(figname)
plt.xlabel('Cell volume (um^3)')
plt.ylabel('Cell count')
plt.hist(volumes_segspim, bins, alpha=0.5, color='red', label="OpenSegSPIM data")
plt.legend(prop={'size': 10})
plt.hist(volumes_sitk, bins, alpha=0.55, color='yellow', label="SimpleITK data")
plt.legend(prop={'size': 10})
#plt.savefig(figname)
plt.show()
