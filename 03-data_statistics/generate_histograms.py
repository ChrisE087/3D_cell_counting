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

def gen_histogram(seg_file, stat_file, res_dir):
    # Read the data into pandas data frame
    data = pd.read_csv(stat_file, sep='\t', header=0, usecols=['Volume (um^3)']) 

    # Convert the volume column to numpy array
    volumes_segspim = data.values
    
    # Read the nuclei segmentation and calculate the volumes of each nuclei
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
    diameters = []
    # Print the statistics
    for l in stats.GetLabels():
        volumes_sitk.append(stats.GetPhysicalSize(l))
        #print("Label: {0} Size/Volume: {1}".format(l, stats.GetPhysicalSize(l)))
        #print('_____________________________________________________')
        
    volumes_sitk = np.array(volumes_sitk, dtype='float64')
    volumes_sitk = np.sort(volumes_sitk)
    
    # Create a histogram out of the data
    bins_start = 0
    bins_end = 25
    bins_stepsize = 1
    seg_filename = seg_file.split('\\')
    filename = "Hist_%d-%d_%s_%s.png" % (bins_start, bins_end, seg_filename[2], seg_filename[4])
    bins = np.arange(bins_start, bins_end, bins_stepsize)
    plt.figure()
    if seg_filename[4].endswith('_OpenSegSPIMResults_'):
        actual_file = seg_filename[4][:-20]
    figname = "%s->%s" % (seg_filename[2], actual_file)
    plt.suptitle(figname)
    plt.xlabel('Zellvolumen (um^3)')
    plt.ylabel('Anzahl Zellen')
    plt.hist(volumes_segspim, bins, alpha=0.75, color='red', label="OpenSegSPIM Zählung")
    plt.legend(prop={'size': 10})
    plt.hist(volumes_sitk, bins, alpha=0.5, color='yellow', label="Eigene Zählung")
    plt.legend(prop={'size': 10})
    fig_path = os.path.join(res_dir, filename)
    plt.savefig(fig_path, dpi=300)
    #plt.show()

path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Daten'))

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
                            stat_file = os.path.join(res_dir, 'Nuclei_measurement_results.txt')
                            gen_histogram(seg_file, stat_file, res_dir)