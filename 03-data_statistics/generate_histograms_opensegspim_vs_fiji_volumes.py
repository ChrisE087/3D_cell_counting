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

def gen_histogram(seg_file_opensegspim, segfile_fiji, hist_name, res_dir):
    
    # Read the nuclei segmentation and calculate the volumes of each nuclei
    raw_data_opensegspim, header_opensegspim = nrrd.read(seg_file_opensegspim) #XYZ
    raw_data_fiji, header_fiji = nrrd.read(seg_file_fiji) #XYZ
    
    # Transpose the numpy array from XYZC to CZYX for the use with SimpleITK
    raw_data_opensegspim = np.transpose(raw_data_opensegspim, axes=[2,1,0]) # ZYX
    raw_data_fiji = np.transpose(raw_data_fiji, axes=[2,1,0]) # ZYX
    
    # Make a SimpleITK out of the numpy array
    image_opensegspim = sitk.GetImageFromArray(raw_data_opensegspim, isVector=False) # XYZ
    image_fiji = sitk.GetImageFromArray(raw_data_fiji, isVector=False) # XYZ
    image_opensegspim.SetSpacing(header_opensegspim.get('spacings'))
    image_fiji.SetSpacing(header_opensegspim.get('spacings')) # Same spacings as in opensegspim header
    
    # Get The Connected Components of the volume image. All intensities greater 
    # than 0 are taken into account for the labeling
    cc_opensegspim = sitk.ConnectedComponent(image_opensegspim>0)
    cc_fiji = sitk.ConnectedComponent(image_fiji>0)
    
    # Calculate the statitics of the labeled regions
    stats_opensegspim = sitk.LabelIntensityStatisticsImageFilter()
    stats_opensegspim.Execute(cc_opensegspim, image_opensegspim)
    stats_fiji = sitk.LabelIntensityStatisticsImageFilter()
    stats_fiji.Execute(cc_fiji, image_fiji)
    
    volumes_opensegspim = []
    # Print the statistics
    for l in stats_opensegspim.GetLabels():
        volumes_opensegspim.append(stats_opensegspim.GetPhysicalSize(l))
        #print("Label: {0} Size/Volume: {1}".format(l, stats.GetPhysicalSize(l)))
        #print('_____________________________________________________')     
    volumes_opensegspim = np.array(volumes_opensegspim, dtype='float64')
    volumes_opensegspim = np.sort(volumes_opensegspim)
    
    volumes_fiji = []
    # Print the statistics
    for l in stats_fiji.GetLabels():
        volumes_fiji.append(stats_fiji.GetPhysicalSize(l))
        #print("Label: {0} Size/Volume: {1}".format(l, stats.GetPhysicalSize(l)))
        #print('_____________________________________________________')     
    volumes_fiji = np.array(volumes_fiji, dtype='float64')
    volumes_fiji = np.sort(volumes_fiji)
    
    # Create a histogram out of the data
    bins_start = 0
    bins_end = 25
    bins_stepsize = 1
    seg_filename = seg_file_opensegspim.split('\\')
    filename = "Hist_%d-%d_%s_%s.png" % (bins_start, bins_end, seg_filename[2], hist_name)
    bins = np.arange(bins_start, bins_end, bins_stepsize)
    plt.figure()
    if seg_filename[4].endswith('_OpenSegSPIMResults_'):
        actual_file = seg_filename[4][:-20]
    figname = "%s->%s" % (seg_filename[2], actual_file)
    plt.suptitle(figname)
    plt.xlabel('Zellvolumen (um^3)')
    plt.ylabel('Anzahl Zellen')
    plt.hist(volumes_opensegspim, bins, alpha=0.5, color='red', label="Zellvolumen OpenSegSPIM")
    plt.hist(volumes_fiji, bins, alpha=0.5, color='yellow', label="Zellvolumen Fiji")
    plt.legend(prop={'size': 10})
    fig_path = os.path.join(res_dir, filename)
    plt.savefig(fig_path, dpi=300)
    #plt.show()

path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', '..', 'Daten'))

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
                            seg_file_opensegspim = os.path.join(res_dir, 'Nucleisegmentedfill.nrrd')
                            seg_file_fiji = os.path.join(res_dir, 'Nucleisegmentedfill2r.nrrd')
                            gen_histogram(seg_file_opensegspim, seg_file_fiji, 'OpenSegSPIM_vs_Fiji_Segmentation', res_dir)
                            