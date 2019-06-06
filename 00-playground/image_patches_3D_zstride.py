import numpy as np
import tensorflow as tf

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
vol_in = gen_volume2(6,6,6)
#vol_in = gen_volume3(6,6,6,3)

# Expand the dimension for batches
vol_in_exp = np.expand_dims(vol_in, axis=0)

# Expand dimension for depth
vol_in_exp = np.expand_dims(vol_in_exp, axis=4)

# Extract 4 patches of 6x3x3 (ZYX)
with tf.Session() as sess:
    t = tf.extract_volume_patches(vol_in_exp, ksizes=[1, 3, 3, 3, 1], 
                                  strides=[1, 3, 3, 3, 1], 
                                  padding='VALID').eval()
    print(t)

# Reduce the dimensions of batches
patches = t[0,:,:,:,:]

# Extract the patches
z1y1x1 = patches[0,0,0,:]
z1y1x2 = patches[0,0,1,:]
z1y2x1 = patches[0,1,0,:]
z1y2x2 = patches[0,1,1,:]
z2y1x1 = patches[1,0,0,:]
z2y1x2 = patches[1,0,1,:]
z2y2x1 = patches[1,1,0,:]
z2y2x2 = patches[1,1,1,:]

# Reshape the patches to its volumes (ZYX)
z1y1x1 = np.reshape(z1y1x1, (3,3,3))
z1y1x2 = np.reshape(z1y1x2, (3,3,3))
z1y2x1 = np.reshape(z1y2x1, (3,3,3))
z1y2x2 = np.reshape(z1y2x2, (3,3,3))
z2y1x1 = np.reshape(z2y1x1, (3,3,3))
z2y1x2 = np.reshape(z2y1x2, (3,3,3))
z2y2x1 = np.reshape(z2y2x1, (3,3,3))
z2y2x2 = np.reshape(z2y2x2, (3,3,3))

# Build the input volume
a = np.concatenate((np.concatenate((z1y1x1, z2y1x1), axis=0), np.concatenate((z1y1x2, z2y1x2), axis=0)), axis=2)
b = np.concatenate((np.concatenate((z1y2x1, z2y2x1), axis=0), np.concatenate((z1y2x2, z2y2x2), axis=0)), axis=2)
vol_out = np.concatenate((a,b), axis=1)
print(vol_in==vol_out)
