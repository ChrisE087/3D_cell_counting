import numpy as np

def gen_volume(z_dim, y_dim, x_dim):
    vol = np.zeros(shape=(z_dim, y_dim, x_dim), dtype='int', order='C')
    for z in range(z_dim):
        for y in range(y_dim):
            for x in range(x_dim):
                string = ('%s%s%s' % (z+1, y+1, x+1))
                concat = int(string)
                vol[z,y,x] = concat
    return vol

# Set volume dimensions
x_dim = 3 # columns
y_dim = 2 # rows
z_dim = 2 # depth

# Create volume
vol = gen_volume(z_dim, y_dim, x_dim)

# Flatten the volume
flat = vol.flatten()

# Reconstruct the volume
vol = np.reshape(vol, (z_dim, y_dim, x_dim))
