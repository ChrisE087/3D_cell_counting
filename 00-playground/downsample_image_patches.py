import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import nrrd

# Load the data
data, header = nrrd.read('test_data/test_patch.nrrd')
data = data[1,:,:,:]
plt.imshow(data[:,:,15])

# Transpose the numpy array from XYZC to CZYX for the use with SimpleITK
data = np.transpose(data, axes=[2,1,0]) # ZYX

# Make a SimpleITK out of the numpy array
image = sitk.GetImageFromArray(data, isVector=False) # XYZ
orig_w = image.GetWidth()
orig_h = image.GetHeight()
orig_d = image.GetDepth()


rotation_center = (15, 15, 15)
axis = (0,0,1)
angle = 0#np.pi/2.0
translation = (1,1,1)
scale_factor = 1.0
similarity = sitk.Similarity3DTransform(scale_factor, axis, angle, translation, rotation_center)
sitk_interpolator = sitk.sitkBSpline
resampled_image = sitk.Resample(image, image, similarity, 
                                sitk_interpolator, 100.0)
# Make a numpy array out of the image
resampled_image = sitk.GetArrayFromImage(resampled_image)
resampled_image = np.transpose(resampled_image, axes=[2,1,0])


plt.imshow(resampled_image[:,:,4])
















# Define the new width, height and depth
new_w = 16
new_h = 16
new_d = 16
new_size = (new_w, new_h, new_d)

# Setup the Resampling Filter with the new size
dimension = 3
transform = sitk.AffineTransform(dimension)

# Calculate the scale factors for each dimension
sf_w = orig_w/new_w
sf_h = orig_h/new_h
sf_d = orig_d/new_d




transform.Scale((4, sf_h, sf_d)) #  Scale here with scale_factor if necessary
#transform.GetMatrix()
#matrix = np.array(transform.GetMatrix()).reshape((dimension,dimension,dimension))



sitk_interpolator = sitk.sitkBSpline

# # Make a reference image with the new size and spacing

new_size = np.asarray(new_size).tolist()
reference_image = sitk.Image(new_size, sitk.sitkFloat32)
    
# Resample the image
resampled_image = sitk.Resample(image, image, transform, 
                                sitk_interpolator, 100.0)

# Make a numpy array out of the image
resampled_image = sitk.GetArrayFromImage(resampled_image)
resampled_image = np.transpose(resampled_image, axes=[2,1,0])


plt.imshow(resampled_image[:,:,4])
