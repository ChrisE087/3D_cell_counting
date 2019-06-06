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

path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Daten'))


for directory in os.listdir(path_to_data):
    data_dir = os.path.join(path_to_data, directory, 'untreated')
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.nrrd'):
                name = os.path.splitext(filename)[0]
                for subdir in os.listdir(data_dir):
                    res_dir = os.path.join(data_dir, subdir)
                    if(os.path.isdir(res_dir)):
                        if name in subdir:
                            nrrd_file = os.path.join(data_dir, filename)
                            nuclei_file = os.path.join(res_dir, 'OriginalStack.tif')
                            print('Processing file: ', nuclei_file)
                            header = nrrd.read_header(nrrd_file)
                            meta_data, raw_data = bfio.get_tif_stack(filepath=nuclei_file, series=0, depth='t', return_dim_order='XYZC') # XYZC
                            raw_data = raw_data[:,:,:,0]
                            nrrd_seg_file = os.path.join(res_dir, 'OriginalStack.nrrd')
                            nrrd.write(nrrd_seg_file, data=raw_data, header=header, index_order='F')