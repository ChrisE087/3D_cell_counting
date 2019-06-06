import javabridge
import bioformats
import numpy as np
import SimpleITK as sitk
import nrrd
import matplotlib.pyplot as plt
from skimage.external import tifffile as tif

def grayConversion(image):
    grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img

javabridge.start_vm(class_path=bioformats.JARS)

path_to_data = 'test.tif'

# Get XML metadata of complete file
xml_string = bioformats.get_omexml_metadata(path_to_data)
ome = bioformats.OMEXML(xml_string)

# Print the number of images that are in the tif
print(ome.image_count)

# Read the metadata from the first image -> series 0
iome = ome.image(0)
print('Name: ', iome.get_Name())
print('ID: ', iome.get_ID())

# Geth the pixel meta data
dim_order = iome.Pixels.get_DimensionOrder()
pix_type = iome.Pixels.get_PixelType()
x_range = iome.Pixels.get_SizeX()
y_range = iome.Pixels.get_SizeY()
z_range = iome.Pixels.get_SizeZ()
t_range = iome.Pixels.get_SizeT()
c_range = iome.Pixels.get_SizeC()
phy_x = iome.Pixels.get_PhysicalSizeX()
phy_x_unit = iome.Pixels.get_PhysicalSizeXUnit()
phy_y = iome.Pixels.get_PhysicalSizeY()
phy_y_unit = iome.Pixels.get_PhysicalSizeYUnit()
phy_z = iome.Pixels.get_PhysicalSizeZ()
phy_z_unit = iome.Pixels.get_PhysicalSizeZUnit()
plane_count = iome.Pixels.get_plane_count()

print('Dimension Order: ', dim_order)
print('Pixel Type: ', pix_type)
print('X Size: ', x_range)
print('Physical X Size: ', phy_x)
print('Y Size: ', y_range)
print('Physical Y Size: ', phy_y)
print('Z Size: ', z_range)
print('Physical Z Size: ', phy_z)
print('T Size: ', t_range)
print('C Size: ', c_range)

# Get the reader and read every image from the stack of the first image 
# series (series 0)
reader = bioformats.ImageReader(path_to_data)
raw_data = []

# Load the raw data into a numpy array -> Go over the z-range and read every 
# image in the stack
for z in range(z_range):
    raw_image = reader.read(z=z, series=0, rescale=False)
    raw_data.append(raw_image)
raw_data = np.array(raw_data)

# Some image transformation
#raw_data += 50

# Make a SimpleITK image out of the numpy array
sitk_image = sitk.GetImageFromArray(raw_data, isVector=True)
#sitk_image.SetSpacing((phy_x, phy_y, phy_z))

# Write the image
writer = sitk.ImageFileWriter()
writer.SetImageIO("TIFFImageIO")
writer.SetFileName('sitk.tif')
writer.Execute(sitk_image)

javabridge.kill_vm()
