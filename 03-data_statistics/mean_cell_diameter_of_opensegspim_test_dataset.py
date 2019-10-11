import nrrd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

def gen_stats(seg_file):
    # Read the nuclei segmentation
    raw_data, header = nrrd.read(seg_file) #XYZ
    
    # Transpose the numpy array from XYZC to CZYX for the use with SimpleITK
    raw_data = np.transpose(raw_data, axes=[2,1,0]) # ZYX
    
    # Make a SimpleITK out of the numpy array
    image = sitk.GetImageFromArray(raw_data, isVector=False) # XYZ
    #image.SetSpacing(header.get('spacings'))
    
    # Get The Connected Components of the volume image. All intensities greater 
    # than 0 are taken into account for the labeling
    cc = sitk.ConnectedComponent(image>0)
    
    # Calculate the statitics of the labeled regions
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc, image)
    
    # Add the statistics to a table
    table = [['Label', 'Cell volume (um^3)', 'Equivalent Ellipsoid diameter (px^3)']]
    diameters_px = []
    for l in stats.GetLabels():
        ellipsoid_diameters = stats.GetEquivalentEllipsoidDiameter(l)
        diameter_x_px = ellipsoid_diameters[0]/image.GetSpacing()[0]
        diameter_y_px = ellipsoid_diameters[1]/image.GetSpacing()[1]
        diameter_z_px = ellipsoid_diameters[2]/image.GetSpacing()[2]
        mean_ellipsoid_diameter_px = np.mean((diameter_x_px, diameter_y_px, diameter_z_px))
        if not(np.isnan(mean_ellipsoid_diameter_px)):
            diameters_px.append(mean_ellipsoid_diameter_px)
            table.append([l, stats.GetPhysicalSize(l), mean_ellipsoid_diameter_px])
    
    diameters_px = np.array(diameters_px)
    mean_diameter_px = np.mean(diameters_px)
    
    return mean_diameter_px

path_to_seg = os.path.join('..', '..', '..', 'Datensaetze', 'OpenSegSPIM_Beispieldaten', 'Neurosphere', 'Y.nrrd')

# Read the nuclei segmentation
seg_np, seg_header = nrrd.read(path_to_seg)

# Transpose the numpy array from XYZC to CZYX for the use with SimpleITK
seg_np = np.transpose(seg_np, axes=[2,1,0]) # ZYX
    
# Make a SimpleITK out of the numpy array
image = sitk.GetImageFromArray(seg_np, isVector=False) # XYZ

# Get The Connected Components of the volume image. All intensities greater 
# than 0 are taken into account for the labeling
cc = sitk.ConnectedComponent(image>0)

# Calculate the statitics of the labeled regions
stats = sitk.LabelIntensityStatisticsImageFilter()
stats.Execute(cc, image)

# Add the statistics to a table
diameters_px = []

for l in stats.GetLabels():
    ellipsoid_diameters = stats.GetEquivalentEllipsoidDiameter(l)
    diameter_x_px = ellipsoid_diameters[0]/image.GetSpacing()[0]
    diameter_y_px = ellipsoid_diameters[1]/image.GetSpacing()[1]
    diameter_z_px = ellipsoid_diameters[2]/image.GetSpacing()[2]
    mean_ellipsoid_diameter_px = np.mean((diameter_x_px, diameter_y_px, diameter_z_px))
    if not(np.isnan(mean_ellipsoid_diameter_px)):
        diameters_px.append(mean_ellipsoid_diameter_px)
        
print('Mean nuclei diameter: ', np.mean(diameters_px))
