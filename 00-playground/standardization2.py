import numpy as np

# Sample data
a = np.random.randint(low=0, high=255, size=32*32*32).astype(np.float64)
print(np.mean(a))
print(np.std(a))

# Calculate the mean
a_sum = 0
for i in range(np.size(a)):
    a_sum = a_sum + a[i]
mean = a_sum/np.size(a)

# Calculate the standard deviation
std = 0
numerator = 0
for i in range(np.size(a)):
    tmp = a[i] - mean
    numerator = numerator + np.square(tmp)
std = np.sqrt(numerator / (np.size(a) - 1))

# Standardizate each voxel
a_std = np.zeros_like(a)

for i in range(np.size(a)):
    a_std[i] = (a[i] - mean) / std
    
print(np.mean(a_std))
print(np.std(a_std))