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
data, header = nrrd.read('test_data/Testbild.nrrd') #XY
data = np.transpose(data, axes=(1,0))
plt.imshow(data)

# Make a SimpleITK out of the numpy array
image = sitk.GetImageFromArray(data, isVector=False) # XY

x_scale, y_scale = 0.5, 1

affine = sitk.AffineTransform(2)
affine.Scale((x_scale, y_scale))
resampled = resample(image, image, affine)

image = sitk.GetArrayFromImage(resampled)

plt.imshow(image)
