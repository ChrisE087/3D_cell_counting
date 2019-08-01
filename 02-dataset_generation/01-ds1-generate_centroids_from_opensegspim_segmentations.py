import sys
sys.path.append("..")
import os
import javabridge
import bioformats
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import nrrd
from tools import image_io as bfio
from tools import image_processing as impro

path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Daten'))

for directory in os.listdir(path_to_data):
    data_dir = os.path.join(path_to_data, directory, 'untreated')
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.nrrd'):
                name = os.path.splitext(filename)[0]
                #print('Actual file: ', name)
                for subdir in os.listdir(data_dir):
                    res_dir = os.path.join(data_dir, subdir)
                    if(os.path.isdir(res_dir)):
                        if name in subdir:
                            #nrrd_file = os.path.join(data_dir, filename)
                            seg_file = os.path.join(res_dir, 'Nucleisegmentedfill.nrrd')
                            print('Processing file: ', seg_file)
                            seg_raw, seg_header = nrrd.read(seg_file) #XYZ
                            spacings = seg_header.get('spacings')
                            centroids, statistics = impro.get_centroids(seg_raw, spacings, 1.)
                            nrrd_centroids_file = os.path.join(res_dir, 'centroids.nrrd')
                            nrrd.write(nrrd_centroids_file, data=centroids, header=seg_header, index_order='F')
                            print('Cells: ', np.sum(centroids))