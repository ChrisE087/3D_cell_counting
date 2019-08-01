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

# Start the Java VM
#javabridge.start_vm(class_path=bioformats.JARS)

table = [['File', 'Original number of z-slices', 'Own number of z-slices', 'OpenSegSPIM number of z-slices']]

#path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Daten', '24h', 'untreated'))
path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Daten'))


for directory in os.listdir(path_to_data):
    data_dir = os.path.join(path_to_data, directory, 'untreated')
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            if item.endswith('.tif'):
                tif_name = os.path.splitext(item)
                path_to_tif = os.path.join(data_dir, item)
                path_to_nrrd = os.path.join(data_dir, tif_name[0]+".nrrd")
                print('Processing image: ', path_to_tif)
                orig_meta_data, orig_raw_data = bfio.get_tif_stack(filepath=path_to_tif, series=0, depth='z', return_dim_order='XYZC') # XYZC
                own_raw_data, own_meta_data = nrrd.read(path_to_nrrd)
                #print('Original z: ', orig_raw_data.shape[2], 'Own z: ', own_raw_data.shape[2])
            abspath = os.path.join(data_dir, item)
            if(os.path.isdir(abspath)):
                tiffile = os.path.splitext(path_to_tif)
                tiffile = tiffile[0].split('\\')
                tiffile = tiffile[-1]
                if(tiffile in abspath):
                    path_to_original_stack = os.path.join(abspath, 'OriginalStack.tif')
                    #print('Correspondig OpenSegSPIM data: ', path_to_original_stack)
                    oss_meta_data, oss_raw_data = bfio.get_tif_stack(filepath=path_to_original_stack, series=0, depth='t', return_dim_order='XYZC') # XYZC
                    #print('Own resized z: ', own_raw_data.shape[-2], 'Segspim resized z: ', oss_raw_data.shape[-2])
                    #print('############################################')
                    if(own_raw_data.shape[2] != oss_raw_data.shape[2]):
                        print(path_to_original_stack, ' has a different z-size')
            splitpath = abspath.split(os.path.sep)
            curr_file = splitpath[2]+'->'+tif_name[0]
            table.append((curr_file, orig_raw_data.shape[2], own_raw_data.shape[2], oss_raw_data.shape[2]))

with open('z_sizes.csv','w') as file:
    for item in table:
        line = "%s \t %s \t %s \t %s\n" %(item[0], item[1], item[2], item[3])
        file.write(line)