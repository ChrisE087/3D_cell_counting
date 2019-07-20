import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import keras
import os
import time
import datetime
import tensorflow as tf
import nrrd
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tools.cnn import CNN
from tools import datagen
from tools import image_processing as impro
from tools import datatools
import SimpleITK as sitk

path_to_nuclei = os.path.join('..', '..', '..', 'Daten', '48h', 'untreated', 'C1-untreated_4.1_8_bit.nrrd')
path_to_seg = os.path.join('..', '..', '..', 'Daten', '48h', 'untreated', 'C1-untreated_4.1_OpenSegSPIMResults_', 'Nucleisegmentedfill2r.nrrd')

nuclei, nuclei_header = nrrd.read(path_to_nuclei)
seg, seg_header = nrrd.read(path_to_seg)

spacings = seg_header.get('space directions')
spacings = [spacings[0,0], spacings[1,1], spacings[2,2]]
spacings = np.array(spacings)
#centroids, statistics = impro.get_centroids(seg, spacings, 1.)

#nuclei = np.zeros(shape=(300, 200, 500)) #XYZ
#seg = np.zeros_like(nuclei)
#seg[150:170, 90:105, 200:230] = 1

raw_data = np.copy(seg)
excluded_volume_size = 1.
spacings = np.array([0.5681, 0.5681, 0.5681])

# Transpose the numpy array from XYZC to CZYX for the use with SimpleITK
raw_data = np.transpose(raw_data, axes=[2,1,0]) # ZYX

# Make a SimpleITK out of the numpy array
image = sitk.GetImageFromArray(raw_data, isVector=False) # XYZ
image.SetSpacing(spacings)
print(image.GetSpacing())

# Get The Connected Components of the volume image. All intensities greater 
# than 0 are taken into account for the labeling
cc = sitk.ConnectedComponent(image>0)

# Calculate the statitics of the labeled regions
statistics = sitk.LabelIntensityStatisticsImageFilter()
statistics.Execute(cc, image)

# Make a volume image for the centroids
centroids = np.zeros_like(raw_data)

# Make a white dot [f(x,y,z)=1] at each centroid of the label l
for l in statistics.GetLabels():
    phy_size = statistics.GetPhysicalSize(l)
    
    # Notice only volumes smaller than a specific volume size
    if phy_size > excluded_volume_size:
        # Get the centroid coordinates of the actual label l
        centroid_coords = statistics.GetCenterOfGravity(l) #XYZ
        centroid_coords = centroid_coords/spacings
        
        # Round the centroid coordinates to int
        centroid_coords = np.floor(centroid_coords).astype(int)
        
        # Make a dot in the result volume at the centroid coordinats
        centroids[centroid_coords[2], centroid_coords[1], centroid_coords[0]] = 1 #ZYX
    
# Transpose the result volume back to XYZ
centroids = np.transpose(centroids, (2, 1, 0)) # XYZ

nrrd.write('test_seg.nrrd', seg)
nrrd.write('test_center.nrrd', centroids)
