import numpy as np
import nrrd

test_image, header = nrrd.read('test_data/testbild.nrrd')
test_image_stack = np.zeros(shape=(512,512,4), dtype='uint8')

for i in range(4):
    test_image_stack[:,:,i] = test_image
    
nrrd.write('test_image_stack.nrrd', test_image_stack)
