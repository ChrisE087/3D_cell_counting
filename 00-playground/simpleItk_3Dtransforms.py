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
data, header = nrrd.read('test_data/test_image_stack.nrrd') #XYZ
plt.imshow(data[:,:,1])

# Transpose the numpy array from XYZ to ZYX for the use with SimpleITK
data = np.transpose(data, axes=(2,1,0))

# Make a SimpleITK out of the numpy array
image = sitk.GetImageFromArray(data, isVector=False) # XY
w = image.GetHeight()
h = image.GetHeight()
d = image.GetDepth()

# Calculate the scale factors in each dimension
x_scale, y_scale, z_scale = 0.5, 1., 1/128

affine = sitk.AffineTransform(3)
affine.Scale((x_scale, y_scale, z_scale))
resampled = resample(image, image, affine)

image = sitk.GetArrayFromImage(resampled)
image = np.transpose(image, axes=(2,1,0))

plt.imshow(image[:,:,2])
