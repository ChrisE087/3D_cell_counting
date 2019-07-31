import os

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

path_to_data = os.path.join('..', '..', 'Daten')

subdirs1 = get_immediate_subdirectories(path_to_data)

for subdir1 in subdirs1:
    if '24h' in subdir1 or '48h' in subdir1 or '72h' in subdir1:
        # Cultivation period level
        data_dir = os.path.join(path_to_data, subdir1)
        subdirs2 = get_immediate_subdirectories(data_dir)
        for subdir2 in subdirs2:
            # Spheroid data level
            untreated_dir = os.path.join(data_dir, subdir2)
            subdirs3 = get_immediate_subdirectories(untreated_dir)
            for subdir3 in subdirs3:
                # OpensegSPIM results level
                result_dir = os.path.join(untreated_dir, subdir3)
                print(result_dir)
