import SimpleITK as sitk
import numpy as np

# Filenames
inputImageFileName = 'test.tif'
outputImageFileName = 'sitk_image.tif'

# Read TIF-image with SimpleITK
reader = sitk.ImageFileReader()
reader.SetImageIO("TIFFImageIO")
reader.SetFileName(inputImageFileName)
image = reader.Execute()

pixel_id = image.GetPixelID()   # Datatype ID
pixel_id_string = image.GetPixelIDTypeAsString()
image_size = image.GetSize()    # Number of pixels/voxels in each dimension (YXZ)
origin = image.GetOrigin()  # Coordinates of the pixel/voxel with index (0,0,0) in physical units (i.e. mm) (XYZ)
spacing = image.GetSpacing() # Coordinates of the pixel/voxel with index (0,0,0) in physical units (i.e. mm) (XYZ)
direction = image.GetDirection() # Mapping, rotation, between direction of the pixel/voxel axes and physical directions

# Print the image properties
print('Pixel Type: ', pixel_id_string, ' ID: ' , pixel_id)
print('Size: ', image_size)
print('Origin: ', origin)
print('Spacing: ', spacing)
print('Direction ', direction)

# Read the additional image information
all_keys = image.GetMetaDataKeys()
for key in all_keys:
    value = image.GetMetaData(key)
    print(key, ': ',  value)

# Convert the SimpleITK image into an numpy array
npimg = sitk.GetArrayFromImage(image)

c1 = npimg[:, :, :, 0]
c2 = npimg[:, :, :, 1]
c3 = npimg[:, :, :, 2]

# Process the image
#npimg += 50 

# Convert the numpy array to a SimpleITK image
image1 = sitk.GetImageFromArray(npimg, isVector=True)

# Set the metadata
for key in all_keys:
    value = reader.GetMetaData(key)
    image1.SetMetaData(key, value)
    print(key, " : ", value)
    
image1.SetSpacing(spacing)
image1.SetOrigin(origin)
image1.SetDirection(direction)

# Write the image
writer = sitk.ImageFileWriter()
writer.SetFileName(outputImageFileName)
writer.Execute(image1)

javabridge.kill_vm()