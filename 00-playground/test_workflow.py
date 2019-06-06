import sys
sys.path.append("..")
import javabridge
import bioformats
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import nrrd
from tools import image_io as bfio
from tools import image_processing as impro

# Start the Java VM
javabridge.start_vm(class_path=bioformats.JARS)

path_to_data = 'test_data/Nucleisegmentedfill.tif'
depth = 't'     # z or t

# Get a numpy array from the tif stack with the dimension
meta_data, raw_data = bfio.get_tif_stack(filepath=path_to_data,
                                         series=0, depth=depth, 
                                         return_dim_order='XYZC') # XYZC

# Transpose the numpy array from XYZC to CZYX for the use with SimpleITK
raw_data = np.transpose(raw_data, axes=[3,2,1,0]) # CZYX

# Extract the channel -> make for each channel
raw_data = raw_data[0,:,:,:]

# Make a SimpleITK out of the numpy array and set its metadata
image = sitk.GetImageFromArray(raw_data, isVector=False) # XYZ
image.SetOrigin([0.0, 0.0, 0.0])
image.SetDirection(np.identity(3, dtype=np.double).flatten().tolist())
image.SetSpacing((meta_data.get('physical_size_x'), 
                  meta_data.get('physical_size_y'), 
                  meta_data.get('physical_size_z')))
print(image.GetOrigin())
print(image.GetDirection())
print(image.GetSpacing())

# Make isotropic voxels
resampled_image = impro.make_image_isotropic(image)
print(resampled_image.GetOrigin())
print(resampled_image.GetDirection())
print(resampled_image.GetSpacing())

# Get a numpy array from the resampled simpleITK image
np_image = sitk.GetArrayFromImage(resampled_image)

# Transpose the numpy array from ZYX back to to XYZ and plot an image
np_image = np.transpose(np_image, axes=[2,1,0]) # XYZ
plt.imshow(np_image[:,:, 50])

nrrd.write('test_nrrd.nrrd', data=np_image, header=None, index_order='F')
