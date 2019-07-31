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

# Start the Java VM
#javabridge.start_vm(class_path=bioformats.JARS)

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
                            seg_file = os.path.join(res_dir, 'Nucleisegmented2.tif')
                            print('Processing file: ', seg_file)
                            header = nrrd.read_header(nrrd_file)
                            meta_data, raw_data = bfio.get_tif_stack(filepath=seg_file, series=0, depth='t', return_dim_order='XYZC') # XYZC
                            raw_data = raw_data[:,:,:,0]
                            nrrd_seg_file = os.path.join(res_dir, 'Nucleisegmented2.nrrd')
                            nrrd.write(nrrd_seg_file, data=raw_data, header=header, index_order='F')