import nrrd
import matplotlib.pyplot as plt
import numpy as np
import os
file = os.path.join('..', '..', '..', 'Datensaetze', 'dataset_segmentation_fiji_and_mathematica', '24h_C2-untreated_2.3-00000809.nrrd')
file = os.path.join('dataset', 'train2', '24h_C2-untreated_2.2-00000809.nrrd')
#file = os.path.join('..', '..', '..', 'Datensaetze', 'dataset2_density-maps_mathematica', 'val', 'val_NPC1_C3-4-00002169.nrrd')
data, header = nrrd.read(file)
plt.imshow(data[0,15,])
plt.imshow(data[1,15,])
print(np.min(data[1,]))
print(np.max(data[1,]))

#check_densitys = False
#
#path_to_trainset = os.path.join('..', '..', '..', 'Datensaetze', 'dataset2_segmentation_mathematica', 'train')
#train_list = os.listdir(path_to_trainset)
#for i in range(len(train_list)):
#    filepath = os.path.join(path_to_trainset, train_list[i])
#    data, header = nrrd.read(filepath)
#    cellnum = np.sum(data[1,])
#    if check_densitys == True:
#        if cellnum < 25.0:
#            print(filepath)
#    else:
#        minval = np.min(data[1,])
#        maxval = np.max(data[1,])
#        #print('Min: ', minval, ' Max: ', maxval)
#        if maxval != 1.0:
#            print(filepath)
#
#path_to_valset = os.path.join('..', '..', '..', 'Datensaetze', 'dataset2_segmentation_mathematica', 'val')
#val_list = os.listdir(path_to_valset)
#for i in range(len(val_list)):
#    filepath = os.path.join(path_to_valset, val_list[i])
#    data, header = nrrd.read(filepath)
#    cellnum = np.sum(data[1,])
#    if check_densitys == True:
#        if cellnum < 25.0:
#            print(filepath)
#    else:
#        minval = np.min(data[1,])
#        maxval = np.max(data[1,])
#        #print('Min: ', minval, ' Max: ', maxval)
#        if maxval != 1.0:
#            print(filepath)