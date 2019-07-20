import os
import nrrd
import numpy as np
import matplotlib.pyplot as plt

path_to_opensegspim_dataset = os.path.join('..', '..', '..', 'Daten', 'dataset_opensegspim_size32_stride16')
path_to_fiji_dataset = os.path.join('..', '..', '..', 'Daten', 'dataset_fiji_size32_stride16')

cell_nums_opensegspim = []
cnt = 0
for file in os.listdir(path_to_opensegspim_dataset):
    if cnt % 10 == 0:
        print(file)
    data, header = nrrd.read(os.path.join(path_to_opensegspim_dataset, file))
    cell_nums_opensegspim.append(np.sum(data[1,]))
    cnt = cnt+1
    
nrrd.write('cell_nums_opensegspim.nrrd', np.array(cell_nums_opensegspim))
    
cell_nums_fiji = []
for file in os.listdir(path_to_fiji_dataset):
    data, header = nrrd.read(os.path.join(path_to_fiji_dataset, file))
    cell_nums_fiji.append(np.sum(data[1,]))
    
nrrd.write('cell_nums_fiji.nrrd', np.array(cell_nums_fiji))


bins = 50
plt_range = (0, 25)
plt.figure()
plt.suptitle('Verteilung der Zellanzahl')
plt.xlabel('Anzahl Zellen in einem Patch')
plt.ylabel('Anzahl Patches')
plt.hist(cell_nums_opensegspim, bins=bins, range=plt_range, alpha=0.75, color='red', label='OpenSegSPIM Datensatz')
plt.legend(prop={'size': 10})
plt.hist(cell_nums_fiji, bins=bins, range=plt_range, alpha=0.5, color='yellow', label='Fiji Datensatz')
plt.legend(prop={'size': 10})
plt.savefig('distribution_of_cell_numbers_%s_bins.png' % (bins), dpi=300)