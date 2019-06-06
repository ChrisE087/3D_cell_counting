import javabridge
import bioformats
import numpy as np
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

print('Dimension Order: ', dim_order)
print('Pixel Type: ', pix_type)
print('X Size: ', x_range)
print('Y Size: ', y_range)
print('Z Size: ', z_range)
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

# Load the middle part of the stack (for testing with OpenSegSPIM)
raw_data_middle = []
z_middle = int(z_range/2)
z_bottom = z_middle-5
z_top = z_middle+5
z_range_middle = z_top - z_bottom
for z in range(z_range_middle):
    raw_image_middle = reader.read(z=z_bottom+z, t=0, series=0, rescale=False)
    raw_image_middle = grayConversion(raw_image_middle)
    raw_data_middle.append(raw_image_middle)
    #plt.imshow(raw_image_middle)
raw_data_middle = np.array(raw_data_middle)

# Transpose the data for displaying it in Fiji or MITK, so the dimension order
# is XYCZ
raw_data_middle = raw_data_middle.transpose(1,2,0)
raw_data_middle = raw_data_middle.transpose(1, 2, 3, 0)
raw_data_middle = raw_data_middle[..., np.newaxis]

# Save the image stack in NRRD format
filename = 'test5.nrrd'
nrrd.write(filename, raw_data_middle)

# Save the image stack in tif format (must be ZXY)
tif.imsave('teststack.tif', raw_data_middle.astype('uint8'), bigtiff=True)
tif.imshow(raw_data_middle)


# Testplot
#test = raw_data_middle[:,:,:,8,0]
#plt.imshow(test)
#print(raw_data_middle.shape)

javabridge.kill_vm()
