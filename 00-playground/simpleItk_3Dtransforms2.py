import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import nrrd

def resample(image, reference_image, transform):
    interpolator = sitk.sitkBSpline
    default_value = 100.0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


# Load the data
data, header = nrrd.read('test_data/test_image_stack.nrrd') #XYZ -> Nothing to transpose
#data = data[0,]
plt.imshow(data[:,:,2])

#Transpose the numpy array from XYZ to ZYX for the use with SimpleITK
data_t = np.transpose(data, axes=(2,1,0))

# Make a SimpleITK out of the numpy array
image = sitk.GetImageFromArray(data_t, isVector=False) #
width = image.GetWidth()
height = image.GetHeight()
depth = image.GetDepth()
print('Width: ', width)
print('Height: ', height)
print('Depth: ', depth)

# Define the new width, height and depth
new_width = 1024
new_height = 512
new_depth = 8

# Create a reference image
reference_image = sitk.Image((new_width, new_height, new_depth), sitk.sitkUInt8)
reference_image.SetDirection(image.GetDirection())
reference_image.SetOrigin(image.GetOrigin())
print('Width: ', reference_image.GetWidth())
print('Height: ', reference_image.GetHeight())
print('Depth: ', reference_image.GetDepth())


# Calculate the scale factors in each dimension

width_scale = width/new_width
height_scale = height/new_height
depth_scale = depth/new_depth

# Calculate the scale factors in each dimension
#x_scale, y_scale, z_scale = 0.5, 1., 1/128

affine = sitk.AffineTransform(3)
affine.Scale((width_scale, height_scale, depth_scale))
resampled_image = resample(image, reference_image, affine)

resampled_image = sitk.GetArrayFromImage(resampled_image) # ZYX
resampled_image = np.transpose(resampled_image, axes=(2,1,0))

plt.imshow(resampled_image[:,:,2])
