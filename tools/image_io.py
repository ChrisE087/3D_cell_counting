import bioformats
import numpy as np

def get_tif_stack(filepath, series=0, depth='z', return_dim_order='YXZC'):
    """
    This function assumes, that the Javabridge-VM is started and initialized
    with the bioformats jars:
        javabridge.start_vm(class_path=bioformats.JARS)
    To kill the vom use:
        javabridge.kill_vm()
    
    Parameters:
    filepath (string): Path to the multistack tif file
    series: (int): Number of the image series which shuld be read. Default = 0
    time_step: Number of the time step which should be read. Default = 0
    (NOT IMPLEMENTED)
    depth (string): 'z', when the depth is the z-axis or 't', when the depth is the 
    t-axis
    bit_depth (integer): Power of two from the bit-depth of the file to be read.
    return_dim_order (string):  In which order the raw data should be returned. 
    XYZC, YXZC ...

    Returns:
    meta_data (dictionary): Metadata Dictionary of the tif file
    raw_data (numpy-array): Numpy-Array which contains the raw data of the 
    image stack in the tif file
    """
    # Get XML metadata of complete file
    xml_string = bioformats.get_omexml_metadata(filepath)
    ome_xml = bioformats.OMEXML(xml_string)
    
    # Read the metadata from the image -> series 0
    iome = ome_xml.image(series)
    series_count = ome_xml.get_image_count()
    image_name = iome.get_Name()
    image_id = iome.get_ID()
    image_acquisition = iome.get_AcquisitionDate()
    
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
    
    # Make a dictionary out of the metadata
    meta_data = {}
    meta_data['series_count'] = series_count
    meta_data['image_name'] = image_name
    meta_data['image_id'] = image_id
    meta_data['image_acquisition'] = image_acquisition
    meta_data['channel_count'] = ch_count
    meta_data['original_dimension_order'] = dim_order
    meta_data['numpy_array_dimension_order'] = 'XYZC'
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
        # Allocate a numpy array for the given dimension
        raw_data = np.empty([y_range, x_range, z_range, c_range], dtype=np_dtype)
        # Get the reader and read every image from the stack of the first image 
        # series (series 0) in dimension order XYZC
        with bioformats.get_image_reader(key="cached_bf_reader", path=filepath, \
                                         url=None) as reader:
            # Read the stack for every channel
            for c in range(c_range):
                # Read the whole stack
                for z in range(z_range):
                    # Ignore the time-axis
                    raw_image = reader.read(c=c, z=z, t=0, series=None, index=None, \
                                            rescale=False, wants_max_intensity=False, \
                                            channel_names=None, XYWH=None)
                    raw_data[:, :, z, c] = raw_image
            
    if depth == 't':
        # Allocate a numpy array for the given dimension
        raw_data = np.empty([y_range, x_range, t_range, c_range], dtype=np_dtype)
        # Get the reader and read every image from the stack of the first image 
        # series (series 0) in dimension order XYZC
        with bioformats.get_image_reader(key="cached_bf_reader", path=filepath, \
                                         url=None) as reader:
            # Read the stack for every channel
            for c in range(c_range):
                # Read the whole stack
                for t in range(t_range):
                    # Ignore the z-axis
                    raw_image = reader.read(c=c, z=0, t=t, series=None, index=None, \
                                            rescale=False, wants_max_intensity=False, \
                                            channel_names=None, XYWH=None)
                    raw_data[:, :, t, c] = raw_image
    
    # Clear the cache and close the reader            
    bioformats.release_image_reader("cached_bf_reader")
    bioformats.clear_image_reader_cache()
    reader.close()
    
    if return_dim_order == 'XYZC':
        return meta_data, np.transpose(raw_data, axes=(1, 0, 2, 3))
    else: # YXZC
        return meta_data, raw_data
