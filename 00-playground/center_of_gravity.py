import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import SimpleITK as sitk

img1 = Image.open('test_data\object_1.tif')
img2 = Image.open('test_data\object_2.tif')

obj1 = np.array(img1) / 255
obj2 = np.array(img2) / 255

# Calculate the center of the x-coordinate of object 1
numerator1_x = 0
denominator1_x = np.sum(obj1)
for y in range(obj1.shape[0]):
    for x in range(obj1.shape[1]):
        if(obj1[y,x] == 1):
            numerator1_x += x*obj1[y,x]        
c_obj1_x = np.int(np.ceil((numerator1_x / denominator1_x)))


# Calculate the center of the y-coordinate of object 1
numerator1_y = 0
denominator1_y = np.sum(obj1)
for y in range(obj1.shape[0]):
    for x in range(obj1.shape[1]):
        if(obj1[y,x] == 1):
            numerator1_y += y*obj1[y,x]        
c_obj1_y = np.int(np.ceil((numerator1_y / denominator1_y)))

obj1_1 = obj1
obj1_1[c_obj1_y, c_obj1_x] = 3

plt.figure()
plt.imshow(obj1_1)


# Calculate the center of the x-coordinate of object 1
numerator2_x = 0
denominator2_x = np.sum(obj1)
for y in range(obj2.shape[0]):
    for x in range(obj2.shape[1]):
        if(obj2[y,x] == 1):
            numerator2_x += x*obj2[y,x]        
c_obj2_x = np.int(np.ceil((numerator2_x / denominator2_x)))


# Calculate the center of the y-coordinate of object 1
numerator2_y = 0
denominator2_y = np.sum(obj1)
for y in range(obj2.shape[0]):
    for x in range(obj2.shape[1]):
        if(obj2[y,x] == 1):
            numerator2_y += y*obj2[y,x]        
c_obj2_y = np.int(np.ceil((numerator2_y / denominator2_y)))

obj2_1 = obj2
obj2_1[c_obj2_y, c_obj2_x] = 3

plt.imshow(obj2_1)
