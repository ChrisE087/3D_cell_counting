import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import nrrd

# Kernel width
sigma = 1.0 

# Build the coordinate arrays from -3 to 3 -- make sure they contain 0!
x = np.arange(-3,4,1)
y = np.arange(-3,4,1)
z = np.arange(-3,4,1)
xx, yy, zz = np.meshgrid(x,y,z)

# Build gaussian normal distribution kernel
norm_const = np.float64(1/((2*np.pi)**(3/2)*sigma**3))
kernel = norm_const*np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))

# plot the kernel
for z in range(kernel.shape[2]):
    plt.figure()
    plt.imshow(kernel[:,:,z])

# Make some sample data
data = np.zeros((51,51,51))
data[26,26,0] = 1.
data[5,5,26] = 1.
data[4,5,26] = 1.

# Convolve the sample data with the kernel
filtered = signal.convolve(data, kernel, mode="same")

# plot the result
for z in range(filtered.shape[2]):
    plt.figure()
    plt.imshow(filtered[:,:,z])

# Count the number of Gaussian kernels by integrating over the volume
total_num = np.sum(filtered)

# Round up the integration
total_num = np.ceil(total_num)
    
# Save the volume and view it with Fiji    
nrrd.write('filtered.nrrd', filtered)

