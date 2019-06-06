import numpy as np
import os
import matplotlib.pyplot as plt
import random
import nrrd

def train_val_test_split(data_list, train_split, val_split, test_split, shuffle=True):
    # Shuffle the dataset
    if shuffle == True:
        random.shuffle(data_list)
        
    if (train_split + val_split + test_split) != 1.:
        print("ERROR: train_split, val_split and test_split must sum to one!")
        return
    
    # Calculate the list beginnings and endings of the list slices
    train_begin = 0
    train_end = int(np.ceil(train_split*len(data_list)))  
    val_begin = train_end
    val_end = val_begin+int(np.ceil(val_split*len(data_list)))
    test_begin = val_end
    test_end = len(data_list)
    
    # Split the dataset
    train = data_list[train_begin:train_end]
    val = data_list[val_begin:val_end]
    test = data_list[test_begin:test_end]
    
    return train, val, test
    

dataset_path = os.path.join('..','..','Daten','dataset')

# Save all files in a list
files = os.listdir(dataset_path)

# Split the files into cultivation period
data_24h = []
data_48h = []
data_72h = []

for element in filter(lambda element: '24h' in element, files):
    data_24h.append(element)
    
for element in filter(lambda element: '48h' in element, files):
    data_48h.append(element)
    
for element in filter(lambda element: '72h' in element, files):
    data_72h.append(element)
    
# Plot a random patch
patch = random.choice(data_24h)
patch = os.path.join(dataset_path, patch)
patch, header = nrrd.read(patch)
plt.imshow(patch[0,:,:,8])
plt.imshow(patch[1,:,:,8])


# Split into train validation and test data
train_24h, val_24h, test_24h = train_val_test_split(data_24h, 0.7, 0.15, 0.15, shuffle=True)
train_48h, val_48h, test_48h = train_val_test_split(data_48h, 0.7, 0.15, 0.15, shuffle=True)
train_72h, val_72h, test_72h = train_val_test_split(data_72h, 0.7, 0.15, 0.15, shuffle=True)

# Concatenate the training, validation and test data
train = train_24h + train_48h + train_72h
val = val_24h + val_48h + val_72h
test = test_24h + test_48h + test_72h

# Shuffle the data
random.shuffle(train)
random.shuffle(val)
random.shuffle(test)