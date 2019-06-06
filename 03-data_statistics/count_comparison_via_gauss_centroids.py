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

def count_txt_lines(txt_file):
    with open(txt_file) as f:
        for i, l in enumerate(f):
            pass
    return i+1

path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Daten'))

count_results_file = os.path.join(path_to_data, 'cellcounts.csv')
with open(count_results_file, 'w') as cf:
    cf.write('File \t Gauss centroid counts \t OpenSegSPIM Segmentation cell count \t Absolute difference \t Percentual difference\n')
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
                                cent_file = os.path.join(res_dir, 'gauss_centroids.nrrd')
                                print('Processing file: ', cent_file)
                                gauss_centroids, header = nrrd.read(cent_file) #XYZ
                                gauss_cell_count = np.sum(gauss_centroids)
                                stat_file = os.path.join(res_dir, 'Nuclei_measurement_results.txt')
                                line_count = count_txt_lines(stat_file)
                                diff = gauss_cell_count - line_count
                                percent = np.abs(diff*100/line_count)
                                line = "%s \t %d \t %d \t %d \t %f \n" %(res_dir, gauss_cell_count, line_count, diff, percent)
                                cf.write(line)