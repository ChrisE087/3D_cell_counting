import os
import nrrd
import numpy as np


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
    
def get_files_in_directory(a_dir):
    files = [f for f in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, f))]
    return files

table = [['Name', 'Minimum value', 'Maximum value', 'mean_val', 'std_val']]

path_to_data = os.path.join('..', '..', 'Daten')
subdirs1 = get_immediate_subdirectories(path_to_data)

for subdir1 in subdirs1:
    if '24h' in subdir1 or '48h' in subdir1 or '72h' in subdir1:
        print('-------------')
        # Cultivation period level
        data_dir = os.path.join(path_to_data, subdir1)
        subdirs2 = get_immediate_subdirectories(data_dir)
        for subdir2 in subdirs2:
            # Spheroid data level
            untreated_dir = os.path.join(data_dir, subdir2)
            subdirs3 = get_immediate_subdirectories(untreated_dir)
            spheroid_files = get_files_in_directory(untreated_dir)
            
            # Get all spheroid files in the current directory
            for spheroid in spheroid_files:
                spheroid_name, ext = os.path.splitext(spheroid)
                spheroid = os.path.join(untreated_dir, spheroid)
                if ext == '.nrrd':
                    print('Current Spheroid: ', spheroid)
                    for subdir3 in subdirs3:
                        # OpensegSPIM results level
                        if spheroid_name in subdir3:
                            result_dir = os.path.join(untreated_dir, subdir3)
                            centroids_file = os.path.join(result_dir, 'gauss_centroids.nrrd')
                            print('Corresponding Centroid: ', centroids_file)
                            Y, Y_header = nrrd.read(centroids_file) #XYZ
                            title = centroids_file.split(os.path.sep)[3] + '->' + spheroid_name
                            min_val = np.min(Y)
                            max_val = np.max(Y)
                            mean_val = np.mean(Y)
                            std_val = np.std(Y)
                            table.append([title, min_val, max_val, mean_val, std_val])
                            
with open('min_max_mean_std.txt','w') as file:
    for item in table:
        line = "%s \t %s \t %s \t %s \t %s\n" %(item[0], item[1], item[2], item[3], item[4])
        file.write(line)