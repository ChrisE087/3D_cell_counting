import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import nrrd

def resample_3d_image(sitk_image, sitk_dtype, new_width, new_height, new_depth):
    width = sitk_image.GetWidth()
    height = sitk_image.GetHeight()
    depth = sitk_image.GetDepth()
    print('Old Width: ', width)
    print('Old Height: ', height)
    print('Old Depth: ', depth)
    # Create a reference image
    reference_image = sitk.Image((new_width, new_height, new_depth), sitk_dtype)
    reference_image.SetDirection(image.GetDirection())
    reference_image.SetOrigin(image.GetOrigin())
    print('New Width: ', reference_image.GetWidth())
    print('New Height: ', reference_image.GetHeight())
    print('New Depth: ', reference_image.GetDepth())
    # Calculate the scale factors in each dimension
    width_scale = width/new_width
    height_scale = height/new_height
    depth_scale = depth/new_depth
    affine = sitk.AffineTransform(3)
    affine.Scale((width_scale, height_scale, depth_scale))
    # Resize the image via affine transform and resampling
    interpolator = sitk.sitkBSpline
    default_value = 100.0
    resized_image = sitk.Resample(sitk_image, reference_image, affine,
                         interpolator, default_value)
    return resized_image
    


# Load the data
data, header = nrrd.read('test_data/test_patch.nrrd') #XYZ
data = data[0,]
#plt.imshow(data[:,:,15])

#Transpose the numpy array from XYZ to ZYX for the use with SimpleITK
data_t = np.transpose(data, axes=(2,1,0))

# Make a SimpleITK out of the numpy array
image = sitk.GetImageFromArray(data_t, isVector=False) #

# Resample the image
resized_image = resample_3d_image(image, sitk.sitkUInt8, 55, 55, 55)

resized_image = sitk.GetArrayFromImage(resized_image) # ZYX
resized_image = np.transpose(resized_image, axes=(2,1,0))
nrrd.write('lalala.nrrd', resized_image)

#plt.imshow(resampled_image[:,:,2])
