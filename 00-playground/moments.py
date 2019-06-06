import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import SimpleITK as sitk

def calculate_centroids_self(img):
    num = 0
    sum_y = 0
    sum_x = 0
    
    # Calculate the centroids
    for y in range (img.shape[0]):
        for x in range (img.shape[1]):
            if(img[y, x] == 1):
                num +=1
                sum_y += y
                sum_x += x
                
    c_y = 1/num*sum_y
    c_x = 1/num*sum_x
    
    # Or
    c_x = np.sum(np.where(img)[1])/np.where(img)[1].shape[0]
    c_y = np.sum(np.where(img)[0])/np.where(img)[0].shape[0]
    
    c_x = np.int(np.ceil(c_x))
    c_y = np.int(np.ceil(c_y))
    
    return c_y, c_x

def calculate_centroids_sitk(img):
    img = sitk.GetImageFromArray(img)
    cc = sitk.ConnectedComponent(img>0)
    statistics = sitk.LabelIntensityStatisticsImageFilter()
    statistics.Execute(cc, img)
    for l in statistics.GetLabels():
        centroid_coords = statistics.GetCenterOfGravity(l)
        centroid_coords = np.ceil(centroid_coords).astype(int)
    return centroid_coords
    

img1 = Image.open('test_data\object_1.tif')
img2 = Image.open('test_data\object_2.tif')

obj1 = np.array(img1) / 255
obj2 = np.array(img2) / 255

c1_y, c1_x = calculate_centroids_self(obj1)
c2_y, c2_x = calculate_centroids_self(obj2)

obj1_1 = obj1
obj1_1[c1_y, c1_x] = 3
plt.imshow(obj1_1)

obj2_1 = obj2
obj2_1[c2_y, c2_x] = 3

plt.imshow(obj1_1, cmap='inferno')
plt.imshow(obj2_1, cmap='inferno')