import javabridge
import bioformats
import numpy as np
import nrrd
import matplotlib.pyplot as plt

javabridge.start_vm(class_path=bioformats.JARS)

path_to_data = 'test_data/Nucleisegmentedfill.tif'
depth = 't'     # z or t

# Get XML metadata of complete file
xml_string = bioformats.get_omexml_metadata(path_to_data)
ome = bioformats.OMEXML(xml_string)

# Print the number of images that are in the tif
print(ome.image_count)

# Read the metadata from the first image -> series 0
iome = ome.image(0)
series_count = ome.get_image_count()
image_name = iome.get_Name()
image_id = iome.get_ID()
image_acquisition = iome.get_AcquisitionDate()

print('Series count: ', series_count)
print('Name: ', image_name)
print('ID: ', image_id)
print('Acquisition Date: ', image_acquisition)

# Geth the pixel meta data from the image
ch_count = iome.Pixels.get_channel_count()
dim_order = iome.Pixels.get_DimensionOrder()
pic_id = iome.Pixels.get_ID()
phy_x = iome.Pixels.get_PhysicalSizeX()
phy_x_unit = iome.Pixels.get_PhysicalSizeXUnit()
phy_y = iome.Pixels.get_PhysicalSizeY()
phy_y_unit = iome.Pixels.get_PhysicalSizeYUnit()
phy_z = iome.Pixels.get_PhysicalSizeZ()
phy_z_unit = iome.Pixels.get_PhysicalSizeZUnit()
pix_type = iome.Pixels.get_PixelType()
plane_count = iome.Pixels.get_plane_count()
c_range = iome.Pixels.get_SizeC()
t_range = iome.Pixels.get_SizeT()
x_range = iome.Pixels.get_SizeX()
y_range = iome.Pixels.get_SizeY()
z_range = iome.Pixels.get_SizeZ()

print('Channel Count: ', ch_count)
print('Dimension Order: ', dim_order)
print('ID: ', pic_id)
print('Physical X Size: ', phy_x, ' ', phy_x_unit)
print('Physical Y Size: ', phy_y, ' ', phy_y_unit)
print('Physical Z Size: ', phy_z, ' ', phy_z_unit)
print('Pixel Type: ', pix_type)
print('Plane Count: ', plane_count)
print('CTXYZ: ', c_range, t_range, x_range, y_range, z_range)

# Make a dictionary out of the metadata
meta_data = {}
meta_data['series_count'] = series_count
meta_data['image_name'] = image_name
meta_data['image_id'] = image_id
meta_data['image_acquisition'] = image_acquisition
meta_data['channel_count'] = ch_count
meta_data['dimension_order'] = dim_order
meta_data['picture_id'] = pic_id
meta_data['physical_size_x'] = phy_x
meta_data['physical_size_x_unit'] = phy_x_unit
meta_data['physical_size_y'] = phy_y
meta_data['physical_size_y_unit'] = phy_y_unit
meta_data['physical_size_z'] = phy_z
meta_data['physical_size_z_unit'] = phy_z_unit
meta_data['pixel_data_type'] = pix_type
meta_data['plane_count'] = plane_count
meta_data['channel_size'] = c_range
meta_data['time_size'] = t_range
meta_data['x_size'] = x_range
meta_data['y_size'] = y_range
meta_data['z_size'] = z_range

# Get the datatype of each pixel
    if pix_type == 'uint8':
        np_dtype = np.uint8
    if pix_type == 'uint16':
        np_dtype = np.uint16

if depth == 'z':
    # Get the reader and read every image from the stack of the first image 
    # series (series 0) in dimension order XYZC
    raw_data = np.empty([y_range, x_range, z_range, c_range], dtype=np.uint8)
    #raw_data = []
    with bioformats.get_image_reader(key="cached_bf_reader", path=path_to_data, \
                                     url=None) as reader:
        # Read the stack for every channel
        for c in range(c_range):
            # Read the whole stack
            for z in range(z_range):
                # Ignore the time
                raw_image = reader.read(c=c, z=z, t=0, series=None, index=None, \
                                        rescale=False, wants_max_intensity=False, \
                                        channel_names=None, XYWH=None)
                raw_data[:, :, z, c] = raw_image
                
        bioformats.release_image_reader("cached_bf_reader")
        bioformats.clear_image_reader_cache()
        reader.close()
        
if depth == 't':
    # Get the reader and read every image from the stack of the first image 
    # series (series 0) in dimension order XYZC
    raw_data = np.empty([y_range, x_range, t_range, c_range], dtype=np.uint8)
    #raw_data = []
    with bioformats.get_image_reader(key="cached_bf_reader", path=path_to_data, \
                                     url=None) as reader:
        # Read the stack for every channel
        for c in range(c_range):
            # Read the whole stack
            for t in range(t_range):
                # Ignore the time
                raw_image = reader.read(c=c, z=0, t=t, series=None, index=None, \
                                        rescale=False, wants_max_intensity=False, \
                                        channel_names=None, XYWH=None)
                raw_data[:, :, t, c] = raw_image
                
        bioformats.release_image_reader("cached_bf_reader")
        bioformats.clear_image_reader_cache()
        reader.close()
    
    
#for z in range (z_range):
#    img = raw_data[:, :, z, 0]
#    plt.figure()
#    plt.imshow(img)

# Export in NRRD File
header = {"spacings": [phy_x, phy_y, phy_z, 1], "dimension": raw_data.ndim, \
          "type": "uchar", "sizes": [x_range, y_range, z_range, c_range],  \
          "units": ['"microns"', '"microns"', '"microns"', '"1"']}
nrrd.write('test_nrrd.nrrd', data=raw_data, header=header, index_order='F')

javabridge.kill_vm()
