import sys
sys.path.append("..")
import javabridge
import bioformats
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import nrrd
from tools import image_io as bfio

# Start the Java VM
javabridge.start_vm(class_path=bioformats.JARS)

filepath = 'test_data/Nucleisegmentedfill.tif'
depth = 't'

# Get a numpy array from the tif stack with the dimension
meta_data, raw_data = bfio.get_tif_stack(filepath=filepath, 
                                         series=0, depth=depth, 
                                         return_dim_order='XYZC') # XYZ

# Extract the channel
raw_data = raw_data[:,:,:,0]


# Transpose the numpy array from XYZC to CZYX for the use with SimpleITK
raw_data = np.transpose(raw_data, axes=[2,1,0]) # ZYX

# Make a SimpleITK out of the numpy array
image = sitk.GetImageFromArray(raw_data, isVector=False) # XYZ
print('Dimension: ', image.GetDimension())
print('Width: ', image.GetWidth())
print('Height: ', image.GetHeight())
print('Depth: ', image.GetDepth())
print('XYZ: ', image.GetSize())

# Get The Connected Components of the volume image. All intensities greater 
# than 0 are taken into account for the labeling
cc = sitk.ConnectedComponent(image>0)

# Calculate the statitics of the labeled regions
stats = sitk.LabelIntensityStatisticsImageFilter()
stats.Execute(cc, image)

# Print the statistics
for l in stats.GetLabels():
    print("Label: {0} Size/Volume: {1} Center of gravity: {2} Perimeter: {3} \
          Spherical perimeter: {4} Ellipsoid diameter: {5}  Mean: {6}"\
          .format(l, stats.GetPhysicalSize(l), stats.GetCenterOfGravity(l), \
          stats.GetPerimeter(l), stats.GetEquivalentSphericalPerimeter(l), 
          stats.GetEquivalentEllipsoidDiameter(l), stats.GetMean(l),))
    print('_____________________________________________________')
       
# Make a volume image for the centroids
centroids = np.zeros_like(raw_data)

# Make a white dot at each centroid of the label l
for l in stats.GetLabels():
    centroid_coords = stats.GetCenterOfGravity(l) #XYZ
    print(centroid_coords)
    centroid_coords = np.ceil(centroid_coords).astype(int)
    print(centroid_coords)
    centroids[centroid_coords[2], centroid_coords[1], centroid_coords[0]] = 255
    
centroids = np.transpose(centroids, (2, 1, 0)) # XYZ


# Write the data without header to nrrd-file and view it in Fiji 
nrrd.write('centroids.nrrd', data=centroids, header=None, index_order='F')
