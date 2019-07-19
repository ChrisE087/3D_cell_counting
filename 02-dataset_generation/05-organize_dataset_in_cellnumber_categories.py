import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import os
import nrrd
from operator import itemgetter
from shutil import copyfile

path_to_dataset = os.path.join('..', '..', '..', 'Daten', 'dataset_size64_stride32')
files = os.listdir(path_to_dataset)
cell_nums = []

for i in range(len(files)):
    data, header = nrrd.read(os.path.join(path_to_dataset, files[i]))
    cell_nums.append([files[i], np.sum(data[1,])])
    
cell_nums_sorted = sorted(cell_nums, key=itemgetter(1))
cell_num_steps = np.arange(0, 500, 0.1)
step = 0
db = []

cnt0 = 0
for i in range(len(cell_nums_sorted)):
    if cell_nums_sorted[i][1] >= cell_num_steps[step] and cell_nums_sorted[i][1] < cell_num_steps[step+1]:
        #print(i, ' Step ', step, ' in range from ', cell_num_steps[step], ' to ', cell_num_steps[step+1])
        #db.append([cell_num_steps[step], cell_num_steps[step+1], cell_nums_sorted[i][0], cell_nums_sorted[i][1]])
        r = '%05.2f_%s_%05.2f' % (cell_num_steps[step], 'to', cell_num_steps[step+1])
        #r = str(cell_num_steps[step])+'_to_'+str(cell_num_steps[step+1])
        range_dir = os.path.join(path_to_dataset, r)
        if not os.path.exists(range_dir):
            os.makedirs(range_dir)
        src = os.path.join(path_to_dataset, cell_nums_sorted[i][0])
        dst = os.path.join(path_to_dataset, range_dir, cell_nums_sorted[i][0])
        copyfile(src, dst)
    else:
        if step < len(cell_num_steps)-1:
            #print('Range was from ', cell_num_steps[step], ' to ', cell_num_steps[step+1])
            step = step+1
            print(step)
        else:
            print('end')
            break