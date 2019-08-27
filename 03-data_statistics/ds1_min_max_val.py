import sys
sys.path.append("..")
import os
import numpy as np
import nrrd
import matplotlib.pyplot as plt

path_to_data = os.path.join('..', '..', '..', 'Daten')

table = []
table.append(['Spheroid', 'Min', 'Max'])

values = np.array([])

for directory in os.listdir(path_to_data):
    data_dir = os.path.join(path_to_data, directory, 'untreated')
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.nrrd'):
                spheroid_name = os.path.splitext(filename)[0]
                #print('Actual file: ', spheroid_name)
                for subdir in os.listdir(data_dir):
                    res_dir = os.path.join(data_dir, subdir)
                    if(os.path.isdir(res_dir)):
                        if spheroid_name in subdir:
                            nrrd_file = os.path.join(data_dir, filename)
                            gauss_centroids_file = os.path.join(res_dir, 'gauss_centroids_fiji_seg.nrrd')
                            gauss_centroids, header = nrrd.read(gauss_centroids_file) #XYZ
                            min_val = np.min(gauss_centroids[gauss_centroids > 0])
                            max_val = np.max(gauss_centroids)
                            values = np.append(values, gauss_centroids.flatten())
                            # Log the number of cells, min and max in a table
                            spheroid_title = res_dir.split(os.path.sep)[2] + '->' + spheroid_name
                            table.append([spheroid_title, min_val, max_val])
                            
with open('das1-min_max_val.txt','w') as file:
    for item in table:
        line = "%s \t %s \t %s\n" %(item[0], item[1], item[2])
        file.write(line)
                            
                
plt.figure()
plt.title('Voxelwerteverteilung in den Density-Maps')
plt.xlabel('Voxelwerte')
plt.ylabel('Werteanzahl')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
plt.hist(values[values > 0], bins=100, range=(0, 0.5e-3))
plt.show()
