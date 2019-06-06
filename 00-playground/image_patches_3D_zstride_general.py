import numpy as np
import tensorflow as tf

# Some methods to generate a 3d-volume
def gen_volume(z_dim, y_dim, x_dim):
    vol = np.zeros(shape=(z_dim, y_dim, x_dim), dtype='int', order='C')
    for z in range(z_dim):
        for y in range(y_dim):
            for x in range(x_dim):
                vol[z,y,x] = z
    return vol  

def gen_volume2(z_dim, y_dim, x_dim):
    vol = np.zeros(shape=(z_dim, y_dim, x_dim), dtype='int', order='C')
    for z in range(z_dim):
        for y in range(y_dim):
            for x in range(x_dim):
                string = ('%s%s%s' % (z+1, y+1, x+1))
                concat = int(string)
                vol[z,y,x] = concat
    return vol

def gen_volume3(z_dim, y_dim, x_dim, kernel_size):
    vol = np.zeros(shape=(z_dim, y_dim, x_dim), dtype='int', order='C')
    cnt_x = 0
    cnt_y = 0
    for z in range(z_dim):
        for y in range(y_dim):
            for x in range(x_dim):
                print(z,y,x)
                vol[z, y, x] = cnt_y+cnt_x
                if(x+1)%kernel_size == 0:
                    cnt_x += 1
            if(y+1)%kernel_size == 0:
                cnt_y += 1
            cnt_x = cnt_y
        cnt_y = cnt_x = 0
    return vol

# Generate 3D Volume with ZYX-Dimensions of 6x6x6
#vol_in = gen_volume(6,6,6)
vol_in = gen_volume2(33,10,40)
#vol_in = gen_volume3(6,6,6,3)

# Expand the dimension for batches
vol_in_exp = np.expand_dims(vol_in, axis=0)

# Expand dimension for depth
vol_in_exp = np.expand_dims(vol_in_exp, axis=4)

# Specify the kernel-size -> volume-size of the extracted patches
k_size_z = 4 # k_size_planes
k_size_y = 2 # ksize_rows
k_size_x = 5 # ksize_cols

# Specify the strides
stride_z = k_size_z # stride_planes
stride_y = k_size_y # stride_rows
stride_x = k_size_x # stride_cols

# Extract  patches of k_size_z x k_size_y x k_size_x
with tf.Session() as sess:
    t = tf.extract_volume_patches(vol_in_exp, 
                                  ksizes=[1, k_size_z, k_size_y, k_size_x, 1], 
                                  strides=[1, stride_z, stride_y, stride_x, 1], 
                                  padding='VALID').eval()
    print(t)
    
    # Reshape the patches to 3D
    # t.shape[1] -> number of extracted patches in z-direction
    # t.shape[2] -> number of extracted patches in y-direction
    # t.shape[3] -> number of extracted patches in x-direction
    t = tf.reshape(t, [1, t.shape[1], t.shape[2], t.shape[3], 
                       k_size_z, k_size_y, k_size_x]).eval()
    

# Reduce the dimensions of batches
patches = t[0,:,:,:,:]

# Extract the patches and build the volume
for z in range(patches.shape[0]):
    for y in range(patches.shape[1]):
        for x in range(patches.shape[2]):
            # Extract a 3D-patch
            patch = patches[z,y,x,:]
            
            # First x-patch? -> Initialize a volume, else concatenate with the
            # last patch on the x-axis
            if(x==0):
                x_concat = patch
            else:
                x_concat = np.concatenate((x_concat, patch), axis=2)
        # First y-patch? -> Initialize a volume, else concatenate with the
        # last patch on the y-axis
        if(y==0):
            y_concat = x_concat
        else:
            y_concat = np.concatenate((y_concat, x_concat), axis=1)
    # First z-patch? -> Initialize a volume, else concatenate with the
    # last patch on the z-axis
    if(z==0):
        z_concat = y_concat
    else:
        z_concat = np.concatenate((z_concat, y_concat), axis=0)

# The output volume is the over all three axes concatenated patches
vol_out = z_concat

# Elementwise comparison could fail, because vol_out could be slightly smaller
# than volume_in depending on the kernel-size, strides and padding
print(vol_in==vol_out)
            




