import tensorflow as tf
import javabridge 
import bioformats as bf
from xml.etree import ElementTree as eltree
import numpy as np
from matplotlib import pyplot as plt, cm

DEFAULT_DIM_ORDER = 'tzyxc'
def parse_xml_metadata(xml_string, array_order=DEFAULT_DIM_ORDER):
    """Get interesting metadata from the LIF file XML string.
    Parameters
    ----------
    xml_string : string
        The string containing the XML data.
    array_order : string
        The order of the dimensions in the multidimensional array.
        Valid orders are a permutation of "tzyxc" for time, the three
        spatial dimensions, and channels.
    Returns
    -------
    names : list of string
        The name of each image series.
    sizes : list of tuple of int
        The pixel size in the specified order of each series.
    resolutions : list of tuple of float
        The resolution of each series in the order given by
        `array_order`. Time and channel dimensions are ignored.
    """
    array_order = array_order.upper()
    names, sizes, resolutions = [], [], []
    spatial_array_order = [c for c in array_order if c in 'XYZ']
    size_tags = ['Size' + c for c in array_order]
    res_tags = ['PhysicalSize' + c for c in spatial_array_order]
    metadata_root = eltree.fromstring(xml_string.encode('utf-8'))
    for child in metadata_root:
        if child.tag.endswith('Image'):
            names.append(child.attrib['Name'])
            for grandchild in child:
                if grandchild.tag.endswith('Pixels'):
                    att = grandchild.attrib
                    sizes.append(tuple([int(att[t]) for t in size_tags]))
                    resolutions.append(tuple([float(att[t])
                                              for t in res_tags]))
    return names, sizes, resolutions


# Start the Java-Bridge and allocate more memory for the jvm
javabridge.start_vm(class_path=bf.JARS, max_heap_size='8G')

# Path to the lif-file
lif_filepath='test.lif'

# Get an XML string of metadata which contains information about the images
# in the file.
omexml = bf.get_omexml_metadata(path = lif_filepath)

# Set the filepath and setup the bioformats image-reader
filename = 'test.lif'
rdr = bf.ImageReader(filename, perform_init=True)

# Get the names, sizes and resolutions of the images, contained in the stack
names, sizes, resolutions = parse_xml_metadata(omexml)

# Get the total number of images in the stack
total_num = len(names)-1

for i in range (total_num):
    # Get the size of the image
    size = sizes[i]
    # Get the time and z-size
    nt, nz = size[:2]
    # Preallocate a numpy array for the image
    image5d = np.empty(size, np.uint8)
    # Go over all t images
    for t in range(nt):
        # Go over all z images in the stack
        print('t=', t)
        for z in range(nz):
            print('z=', z)
#            # Read the actual image
#            image5d[t, z] = rdr.read(z=z, t=t, series=i, rescale=False)
#            # plot the image
#            plt.imshow(image5d[nt//2, nz//2, :, :, 0], cmap=cm.gray)
#            plt.show()