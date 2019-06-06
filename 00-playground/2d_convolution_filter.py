import numpy as np
from scipy import signal

def oversample_image(image, factor):
    resampled_image = np.zeros((image.shape[0] * factor, image.shape[1] *factor))
    curr_y = 0
    curr_x = 0
    for y in range(resampled_image.shape[0]):
        for x in range(resampled_image.shape[1]):
            if x%factor == 0 and y%factor == 0:
                resampled_image[y, x] = image[curr_y, curr_x]
                curr_x += 1
                print('Eingefügt in spalte ', x)
        curr_x = 0
        if y%factor == 0:
            curr_y += 1
                
    return resampled_image
                

# Create a 2D image
im = [50,80,10,30,12,32,25,30,43]
im = np.array(im, dtype='uint8')
im = im.reshape((3,3))

# Create the new image. It's size is oversampled horizontally and vertically and 
# filled with zeros in between
sampling_factor = 3
resampled_im = oversample_image(im, sampling_factor)

## Create the kernel
k_nn = np.ones((sampling_factor, sampling_factor), dtype='float')

res_nn = signal.convolve2d(resampled_im, k_nn, mode='full')
#res_bilin = signal.convolve2d(resampled_im, k_bilin, mode='full')

i = np.array([1,1,1,1,3,1,1,1,1]).reshape((3,3))
k = np.ones((3,3))
c = signal.convolve(i, k, mode='valid')
