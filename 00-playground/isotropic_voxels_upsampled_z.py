import sys
sys.path.append("..")
import javabridge
import bioformats
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import nrrd
from tools import image_io as bfio

# Start the Java VM
javabridge.start_vm(class_path=bioformats.JARS)

# Get a numpy array from the tif stack with the dimension
meta_data, raw_data = bfio.get_tif_stack(filepath='test_data/test_nucleid_1ch.tif',
                                         series=0, depth='z', 
                                         return_dim_order='XYZC') # XYZC


# Write the data without header to nrrd-file and view it in Fiji 
# -> the z-axis is much shorter
nrrd.write('export\test_nrrd.nrrd', data=raw_data, header=None, index_order='F')

# Transpose the numpy array from XYZC to CZYX for the use with SimpleITK
raw_data = np.transpose(raw_data, axes=[3,2,1,0]) # CZYX

# Extract the channel
raw_data = raw_data[0,:,:,:]

# Make a SimpleITK out of the numpy array
image = sitk.GetImageFromArray(raw_data, isVector=False) # XYZ
print(image.GetDimension())
print(image.GetWidth())
print(image.GetHeight())
print(image.GetDepth())
print(image.GetSize())

# Set the (default) origin, (default) direction cosine matrix and the
# physical spacing (from the metadata)
image.SetOrigin([0.0, 0.0, 0.0])
image.SetDirection(np.identity(3, dtype=np.double).flatten().tolist())
image.SetSpacing((meta_data.get('physical_size_x'), 
                  meta_data.get('physical_size_y'), 
                  meta_data.get('physical_size_z')))
print(image.GetOrigin())
print(image.GetDirection())
print(image.GetSpacing())

# Setup the Resampling Filter with a Nearest Neighbor interpolator and set the 
# direction and origin from the original image. The new output spacing should
# be [1, 1, 1] for isotropic voxels
new_spacing = [1, 1, 1]
resample_filter = sitk.ResampleImageFilter()
resample_filter.SetInterpolator = sitk.sitkNearestNeighbor
resample_filter.SetOutputDirection = image.GetDirection()
resample_filter.SetOutputOrigin = image.GetOrigin()
resample_filter.SetOutputSpacing(new_spacing)

# Calculate the new z size of the resampled image. The xy size should stay the
# same. The following calculation assumes, that the x and y axis are the same 
# size and the dimension order is xyz
orig_size = np.array(image.GetSize())
orig_spacing = np.array(image.GetSpacing())

# Calculate the scaling factor for the z-axis
if orig_spacing[0] == orig_spacing[1]:
    z_scale_factor = orig_spacing[2] / orig_spacing[0]
else:
    print('Error: x-y spacing is not identical')

# Calculate the new image volume size
new_size = orig_size
new_size[2] = orig_size[2]*z_scale_factor
new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
new_size = np.asarray(new_size).tolist() # [int(s) for s in new_size]

# Calculate the new physical size for the axes and update the metadata
old_spacing = image.GetSpacing()
new_phy_z = old_spacing[2]/z_scale_factor
meta_data['physical_size_z'] = new_phy_z
meta_data['z_size'] = new_size[2]

# Make a reference image with the new size and spacing
reference_image = sitk.Image(new_size, sitk.sitkUInt8)
new_spacing = np.array(old_spacing)
new_spacing[2] = new_phy_z
new_spacing = new_spacing.tolist()
reference_image.SetSpacing(new_spacing)
reference_image.GetSpacing()

# Calculate the scaling factor of the affine transform in each direction
scale_factor = 1/z_scale_factor

# Setup the Resampling Filter with the new size and resample
transform = sitk.AffineTransform(3)
transform.Scale((scale_factor, scale_factor, scale_factor))
resampled_image = sitk.Resample(image, reference_image, transform, 
                                sitk.sitkNearestNeighbor, 100.0)
print(resampled_image.GetDimension())
print(resampled_image.GetWidth())
print(resampled_image.GetHeight())
print(resampled_image.GetDepth())
print(resampled_image.GetSize())
print(resampled_image.GetSpacing())

# Get a numpy array from the resampled simpleITK image
np_image = sitk.GetArrayFromImage(resampled_image)

# Transpose the numpy array from ZYX back to to XYZ and plot an image
np_image = np.transpose(np_image, axes=[2,1,0]) # XYZ
plt.imshow(np_image[:,:, 50])

# Write the data without header to nrrd-file and view it in Fiji 
# -> the z-axis is much longer
new_spacing = resampled_image.GetSpacing()
header = {"spacings": [new_spacing[0], new_spacing[1], new_spacing[2]], 
          "dimension": np_image.ndim,
          "type": "uchar", 
          "sizes": [resampled_image.GetWidth(), resampled_image.GetHeight(), 
                    resampled_image.GetDepth()],
          "units": ['"microns"', '"microns"', '"microns"']}
nrrd.write('export/test_nrrd.nrrd', data=np_image, header=header, index_order='F')
