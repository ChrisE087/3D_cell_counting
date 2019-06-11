import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import nrrd

kernel_size = 50

half_kernel_size = int((kernel_size-1)/2)
# Kernel width
sigma = 6

# Build the coordinate arrays from -3 to 3 -- make sure they contain 0!
x = np.arange(-half_kernel_size, half_kernel_size+1, 1)
y = np.arange(-half_kernel_size, half_kernel_size+1, 1)
z = np.arange(-half_kernel_size, half_kernel_size+1, 1)
xx, yy, zz = np.meshgrid(x,y,z)

# Build gaussian normal distribution kernel
norm_const = np.float32(1/((2*np.pi)**(3/2)*sigma**3))
kernel = norm_const*np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2)).astype(np.float32)


a = np.sum(kernel)

plt.imshow(kernel[:,:,12])

nrrd.write('kernel.nrrd', kernel)

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.voxels(kernel, edgecolor='k')


kernel2 = kernel*255*255
print(np.min(kernel2))
