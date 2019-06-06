import SimpleITK as sitk
import numpy as np
from scipy import signal

def make_image_isotropic(simple_itk_image, interpolator, add_to_z=-1):
    """
    This function assumes, that the Javabridge-VM is started and initialized
    with the bioformats jars:
        javabridge.start_vm(class_path=bioformats.JARS)
    To kill the vom use:
        javabridge.kill_vm()
    
    Parameters:
    simple_itk_image (SimpleITK Image): SimpleITK image of one channel, in 
    which the following information should be set:
        - Origin (default is OK)
        - Direction (default is OK)
        - Spacing (necessary for resampling)
    interpolator (String): Interpolator which should be used (nearest_neighbor, 
    linear, bspline)
    add_to_z (int): Number of z-planes which is added to the calculated z-size

    Returns:
    resampled_image (SimpleITK Image): SimpleITK image, in which the voxel 
    size is isotropic.
   """
    
    # Setup the Resampling Filter with a Nearest Neighbor interpolator and set the 
    # direction and origin from the original image. The new output spacing should
    # be [1, 1, 1] for isotropic voxels
    new_spacing = [1, 1, 1]
    #resample_filter = sitk.ResampleImageFilter()
    if interpolator == 'nearest_neighbor':
        #resample_filter.SetInterpolator = sitk.sitkNearestNeighbor
        sitk_interpolator = sitk.sitkNearestNeighbor
    elif interpolator == 'linear':
        #resample_filter.SetInterpolator = sitk.sitkLinear
        sitk_interpolator = sitk.sitkLinear
    elif interpolator == 'bspline':
        #resample_filter.SetInterpolator = sitk.sitkBSpline
        sitk_interpolator = sitk.sitkBSpline
#    resample_filter.SetOutputDirection = simple_itk_image.GetDirection()
#    resample_filter.SetOutputOrigin = simple_itk_image.GetOrigin()
#    resample_filter.SetOutputSpacing(new_spacing)
    
    # Calculate the new z size of the resampled image. The xy size should stay the
    # same. The following calculation assumes, that the x and y axis are the same 
    # size and the dimension order is xyz
    orig_size = np.array(simple_itk_image.GetSize())
    orig_spacing = np.array(simple_itk_image.GetSpacing())
    
    # Calculate the scaling factor for the z-axis
    if orig_spacing[0] == orig_spacing[1]:
        z_scale_factor = orig_spacing[2] / orig_spacing[0]
    else:
        print('Error: x-y spacing is not identical')
        return
    
    # Calculate the new image volume size
    new_size = orig_size
    
    #new_size[2] = orig_size[2]*z_scale_factor
    #new_size[2] = orig_size[2]*z_scale_factor-1 # -1 because the OpenSegSPIM data is rounded
    new_size[2] = orig_size[2]*z_scale_factor+add_to_z
    
    #new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
    new_size = np.asarray(new_size).tolist() # [int(s) for s in new_size]
    
    # Calculate the new physical size for the axes
    old_spacing = simple_itk_image.GetSpacing()
    new_phy_z = old_spacing[2]/z_scale_factor
    
    # Make a reference image with the new size and spacing
    reference_image = sitk.Image(new_size, sitk.sitkUInt8)
    new_spacing = np.array(old_spacing)
    new_spacing[2] = new_phy_z
    new_spacing = new_spacing.tolist()
    reference_image.SetSpacing(new_spacing)
    
    # Calculate the scaling factor of the affine transform in each direction
    # not necessary in this function but necessary in a script? Crazy things
    # happen here. If the scale factor is wrong, the image is zoomed
    #scale_factor = 1/z_scale_factor

    # Setup the Resampling Filter with the new size and resample
    transform = sitk.AffineTransform(3)
    transform.Scale((1, 1, 1)) #  Scale here with scale_factor if necessary
    resampled_image = sitk.Resample(simple_itk_image, reference_image, transform, 
                                    sitk_interpolator, 100.0)
    return resampled_image

def get_centroids(raw_data):
    """
    This function assumes, that the Javabridge-VM is started and initialized
    with the bioformats jars:
        javabridge.start_vm(class_path=bioformats.JARS)
    To kill the vom use:
        javabridge.kill_vm()
    
    Parameters:
    raw_data (Numpy Array): 3D Numpy array of 3D cell segmentations with 
    dimension order XYZ

    Returns:
    centroids (Numpy Array): 3D Numpy array with the same dimension as the input
    parameter np_array in which every centroid of a cell is marked with a voxel
    with the value of 1. Non centroid areas are marked with 0.
   """
   
    # Transpose the numpy array from XYZC to CZYX for the use with SimpleITK
    raw_data = np.transpose(raw_data, axes=[2,1,0]) # ZYX

    # Make a SimpleITK out of the numpy array
    image = sitk.GetImageFromArray(raw_data, isVector=False) # XYZ
    
    # Get The Connected Components of the volume image. All intensities greater 
    # than 0 are taken into account for the labeling
    cc = sitk.ConnectedComponent(image>0)

    # Calculate the statitics of the labeled regions
    statistics = sitk.LabelIntensityStatisticsImageFilter()
    statistics.Execute(cc, image)
    
    # Make a volume image for the centroids
    centroids = np.zeros_like(raw_data)
    
    # Make a white dot [f(x,y,z)=1] at each centroid of the label l
    for l in statistics.GetLabels():
        # Get the centroid coordinates of the actual label l
        centroid_coords = statistics.GetCenterOfGravity(l) #XYZ
        
        # Round the centroid coordinates to int
        centroid_coords = np.ceil(centroid_coords).astype(int)
        
        # Make a dot in the result volume at the centroid coordinats
        centroids[centroid_coords[2], centroid_coords[1], centroid_coords[0]] = 1 #ZYX
        
    # Transpose the result volume back to XYZ
    centroids = np.transpose(centroids, (2, 1, 0)) # XYZ
    
    return centroids, statistics

def convolve_with_gauss(raw_data):
    
    # Kernel width
    sigma = 3 

    # Build the coordinate arrays from -3 to 3 -- make sure they contain 0!
    x = np.arange(-10,11,1)
    y = np.arange(-10,11,1)
    z = np.arange(-10,11,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    
    # Build gaussian normal distribution kernel
    norm_const = np.float64(1/((2*np.pi)**(3/2)*sigma**3))
    kernel = norm_const*np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    
    # Convolve the data with the kernel
    filtered = signal.convolve(raw_data, kernel, mode="same", method='direct')
    
    return filtered

    