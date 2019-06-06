import sys
sys.path.append("..")
import os
import javabridge
import bioformats
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import nrrd
import pandas as pd
from tools import image_io as bfio
from tools import image_processing as impro

def gen_stats(seg_file, res_dir):
    # Read the nuclei segmentation
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
    
    # Add the statistics to a table
    table = [['Label', 'Cell volume (um^3)', 'Equivalent Ellipsoid diameter (um^3)', 'Equivalent Ellipsoid diameter (px^3)']]
    diameters_um = []
    diameters_px = []
    for l in stats.GetLabels():
        ellipsoid_diameters = stats.GetEquivalentEllipsoidDiameter(l)
        diameter_x_px = ellipsoid_diameters[0]/image.GetSpacing()[0]
        diameter_y_px = ellipsoid_diameters[1]/image.GetSpacing()[1]
        diameter_z_px = ellipsoid_diameters[2]/image.GetSpacing()[2]
        mean_ellipsoid_diameter_px = np.mean((diameter_x_px, diameter_y_px, diameter_z_px))
        mean_ellipsoid_diameter_um = np.mean(ellipsoid_diameters)
        if not(np.isnan(mean_ellipsoid_diameter_um)):
            diameters_px.append(mean_ellipsoid_diameter_px)
            diameters_um.append(mean_ellipsoid_diameter_um)
            table.append([l, stats.GetPhysicalSize(l), mean_ellipsoid_diameter_um, mean_ellipsoid_diameter_px])
    
    diameters_px = np.array(diameters_px)
    diameters_um = np.array(diameters_um)
    mean_diameter_px = np.mean(diameters_px)
    mean_diameter_um = np.mean(diameters_um)
    
    # Create a statistics file of the data
    stats_file = os.path.join(res_dir, 'nuclei_stats.txt')
    with open(stats_file,'w') as file:
        for item in table:
            line = "%s \t %s \t %s \t %s\n" %(item[0], item[1], item[2], item[3])
            file.write(line)
    return mean_diameter_px, mean_diameter_um


path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Daten'))
mean_diameters = [['File', 'Mean diameter in px^3', 'Mean diameter in um^3']]

for directory in os.listdir(path_to_data):
    data_dir = os.path.join(path_to_data, directory, 'untreated')
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.nrrd'):
                name = os.path.splitext(filename)[0]
                print('Actual file: ', name)
                for subdir in os.listdir(data_dir):
                    res_dir = os.path.join(data_dir, subdir)
                    if(os.path.isdir(res_dir)):
                        if name in subdir:
                            nrrd_file = os.path.join(data_dir, filename)
                            seg_file = os.path.join(res_dir, 'Nucleisegmentedfill.nrrd')
                            actual_file = nrrd_file.split(os.path.sep)[2]+'->'+name
                            mean_diameter_px, mean_diameter_um = gen_stats(seg_file, res_dir)
                            mean_diameters.append([actual_file, mean_diameter_px, mean_diameter_um])

with open('mean_diameters.txt','w') as file:
    for item in mean_diameters:
        line = "%s \t %s \t %s\n" %(item[0], item[1], item[2])
        file.write(line)