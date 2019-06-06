import SimpleITK as sitk
import numpy as np
from scipy import signal
import tensorflow as tf

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

    # Setup the Resampling Filter with the new size
    transform = sitk.AffineTransform(3)
    transform.Scale((1, 1, 1)) #  Scale here with scale_factor if necessary
    
    # Set the SimpleITK Interpolator
    if interpolator == 'nearest_neighbor':
        sitk_interpolator = sitk.sitkNearestNeighbor
    elif interpolator == 'linear':
        sitk_interpolator = sitk.sitkLinear
    elif interpolator == 'bspline':
        sitk_interpolator = sitk.sitkBSpline
        
    # Resample the image
    resampled_image = sitk.Resample(simple_itk_image, reference_image, transform, 
                                    sitk_interpolator, 100.0)
    return resampled_image

def get_centroids(raw_data, spacings, excluded_volume_size):
    """
    This function assumes, that the Javabridge-VM is started and initialized
    with the bioformats jars:
        javabridge.start_vm(class_path=bioformats.JARS)
    To kill the vom use:
        javabridge.kill_vm()
    
    Parameters:
    raw_data (Numpy Array): 3D Numpy array of 3D cell segmentations with 
    dimension order XYZ
    spacings (Numpy Array): Array of physical Voxel size in dimension order XYZ
    excluded_volumes (float): Volume size in um^3. Centroids for cell volumes
    smaller than this value are not created

    Returns:
    centroids (Numpy Array): 3D Numpy array with the same dimension as the input
    parameter np_array in which every centroid of a cell is marked with a voxel
    with the value of 1. Non centroid areas are marked with 0.
   """
   
    # Transpose the numpy array from XYZC to CZYX for the use with SimpleITK
    raw_data = np.transpose(raw_data, axes=[2,1,0]) # ZYX

    # Make a SimpleITK out of the numpy array
    image = sitk.GetImageFromArray(raw_data, isVector=False) # XYZ
    image.SetSpacing(spacings)
    print(image.GetSpacing())
    
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
        phy_size = statistics.GetPhysicalSize(l)
        
        # Notice only volumes smaller than a specific volume size
        if phy_size > excluded_volume_size:
            # Get the centroid coordinates of the actual label l
            centroid_coords = statistics.GetCenterOfGravity(l) #XYZ
            centroid_coords = centroid_coords/spacings
            
            # Round the centroid coordinates to int
            centroid_coords = np.ceil(centroid_coords).astype(int)
            
            # Make a dot in the result volume at the centroid coordinats
            centroids[centroid_coords[2], centroid_coords[1], centroid_coords[0]] = 1 #ZYX
        
    # Transpose the result volume back to XYZ
    centroids = np.transpose(centroids, (2, 1, 0)) # XYZ
    
    return centroids, statistics

def convolve_with_gauss(raw_data, kernel_size, sigma):
    """
    This function convolves the 3D data in raw_data with a generated Gaussian
    kernel of the size in kernel_size with the required Sigma
    
    Parameters:
    raw_data (Numpy Array): 3D Numpy array with dimension order XYZ
    kernel_size (int): Integer of the kernel size of a cubic convolution kernel
    sigma: Value of Sigma

    Returns:
    filtered (Numpy Array): 3D Numpy array with the same dimension as the input
    parameter raw_data. The return array is the filtered result of raw_data
    convolved with the Gaussian kernel.
   """
    
    half_kernel_size = int((kernel_size-1)/2)
    
    # Kernel width
    sigma = sigma 

    # Build the coordinate arrays from -3 to 3 -- make sure they contain 0!
    x = np.arange(-half_kernel_size, half_kernel_size+1, 1)
    y = np.arange(-half_kernel_size, half_kernel_size+1, 1)
    z = np.arange(-half_kernel_size, half_kernel_size+1, 1)
    xx, yy, zz = np.meshgrid(x,y,z)
    
    # Build gaussian normal distribution kernel
    norm_const = np.float64(1/((2*np.pi)**(3/2)*sigma**3))
    kernel = norm_const*np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    
    # Convolve the data with the kernel
    #filtered = signal.convolve(raw_data, kernel, mode="same", method='direct')
    filtered = signal.fftconvolve(raw_data, kernel, mode="same")
    
    return filtered

def normalize_data(input_data):
    """
    Normalizes the data in input_data, so that the min-value is zero and the
    max value is one.
    
    Parameters:
    input_data (Numpy Array): ND Numpy Array to normalize.
    
    Returns:
    normalized_data (Numpy Array): Normalized data
    max_val (float): Maximum value in input_data
    min_val (float): Minimum value in input_data
    """
    max_val = np.max(input_data)
    min_val = np.min(input_data)
    normalized_data = (input_data - min_val)/(max_val-min_val)
    normalized_data = normalized_data.astype('float')
    return normalized_data, max_val, min_val

def unnormalize_data(normalized_data, max_val, min_val):
    """
    Undos the normalization in normalized_data.
    
    Parameters:
    normalized_data (Numpy Array): ND Numpy Array to undo the normalization.
    max_val (float): Maximum value in the original data.
    min_val (float): Minimum value in the original data.
    
    Returns:
    unnormalized_data (Numpy Array): Original data without normalization.
    """
    normalized_data = normalized_data.astype('float64')
    unnormalized_data = normalized_data*(max_val-min_val)+min_val
    return unnormalized_data

def standardize_data(input_data):
    """
    Standardizates the data in input_data, so it has a mean of zero and a
    unit-variance.
    
    Parameters:
    input_data (Numpy Array): ND Numpy Array to standardize.
    
    Returns:
    standardized_data_data (Numpy Array): Standardizated data.
    mean (float): Mean of the original data.
    sigma (float): Standard deviation of the original data.
    """
    mean = np.mean(input_data)
    sigma = np.std(input_data)
    standardized_data = (input_data-mean)/sigma
    return standardized_data, mean, sigma

def standardize_data_tf(input_data):
    """
    Standardizates the data in input_data, so it has a mean of zero and a
    unit-variance with TensorFlow.
    
    Parameters:
    input_data (Numpy Array): 3D Numpy Array of shape columns x rows x slices 
    (XYZ) to standardize.
    
    Returns:
    standardized_data_data (Numpy Array): Standardizated data.
    """
    if np.size(input_data.shape) != 3:
        print('input_data must be of shape 3 but has shape ', np.size(input_data.shape))
        return
    
    # Transpose the input-data from XYZ to YXZ for use with TensorFlow
    input_data = np.transpose(input_data, axes=(1,0,2))
    
    # Standadize the data with TensorFlow
    sess = tf.Session()
    with sess.as_default():
        std_data = tf.image.per_image_standardization(input_data).eval()
    
    # Transpose back to XYZ
    std_data = np.transpose(std_data, axes=(1,0,2))

    return std_data
    

def unstandardizate_data(input_data, mean, sigma):
    """
    Reverts the standardization of the data in input_data.
    
    Parameters:
    input_data (Numpy Array): ND Numpy Array to unstandardizate.
    mean (float): Mean of the original data.
    sigma (float): Standard deviation of the original data.
    
    Returns:
    unstandardizated_data (Numpy Array): Unstandardizated data.
    """
    unstandardizated_data = input_data*sigma+mean
    return unstandardizated_data

def scale_data(data, factor):
    """
    Scales the input data in data by a factor in factor.
    
    Parameters:
    data (Numpy Array): ND Numpy Array to scale.
    factor (float): Scale factor
    
    Returns:
    scaled_data (Numpy Array): Scaled data with factor.
    """
    scaled_data = data*factor
    scaled_data = scaled_data.astype('uint16')
    return scaled_data

def unscale_data(data, factor):
    """
    Undos the scaling of the input data in data inverse scaling with a factor
    in factor.
    
    Parameters:
    data (Numpy Array): ND Numpy Array to scale.
    factor (float): Scale factor
    
    Returns:
    scaled_data (Numpy Array): Inverse scaled data with factor.
    """
    unscaled_data = data.astype('float')
    unscaled_data = data/factor
    return unscaled_data

def gen_patches(data, patch_slices, patch_rows, patch_cols, stride_slices, 
                stride_rows, stride_cols, input_dim_order='XYZ', padding='VALID'):
    """
    Generates patches of the Numpy Array data of the size patch_slices x 
    patch_rows x patch_cols with stride_slices x stride stride_rows x
    stride_cols.
    
    Parameters:
    data (Numpy Array): Numpy Arrayout of which the patches are generated.
    patch_slices (int): Number of slices (z-size) one patch should have.
    patch_rows (int): Number of rows (y-size) one patch should have.
    patch_cols (int): Number of columns (x-size) one patch should have.
    stride_slices (int): Stride in slice direction (z-direction).
    stride_rows (int): Stride in row direction (y-direction).
    stride_cols (int): Stride in column direction (x-direction).
    input_dim_order (String): String of the dimension order of data. Can be 
    'XYZ' oder 'ZYX'.
    padding (String): String which padding should be used. Can be 'VALID' 
    (no padding) or 'SAME' (with zero-padding).
    
    Returns:
    patches (Numpy Array): Generated Patches of size slice_indice x row_indice
    x column_indice x image_slice x image_row x image_column
    """
    
    # Reorder the dimensions to ZYX
    if input_dim_order == 'XYZ':
        data = np.transpose(data, axes=(2,1,0))
    
    # Check if the data has channels
    if np.size(data.shape) != 3:
        print('WARNING! Function is only meant to be used for data with one channel')
        return
    
    # Expand dimension for depth (number of channels)
    data = data[:,:,:,np.newaxis]
    
    # Expand the dimension for batches
    data = data[np.newaxis,:,:,:,:]
    
    # Extract  patches of size patch_slices x patch_rows x patch_cols
    with tf.Session() as sess:
        t = tf.extract_volume_patches(data, ksizes=[1, patch_slices, patch_rows, patch_cols, 1], 
                                      strides=[1, stride_slices, stride_rows, stride_cols, 1], 
                                      padding=padding).eval(session=sess)
    
        # Reshape the patches to 3D
        # t.shape[1] -> number of extracted patches in z-direction
        # t.shape[2] -> number of extracted patches in y-direction
        # t.shape[3] -> number of extracted patches in x-direction
        t = tf.reshape(t, [1, t.shape[1], t.shape[2], t.shape[3], 
                           patch_slices, patch_rows, patch_cols]).eval(session=sess)
        sess.close()
    
    # Remove the batch dimension
    patches = t[0,:,:,:,:]
    
    # Remove the channel dimension
    #if has_channels == False:
        #patches = t[:,:,:,0]
    
    return patches

def restore_volume(patches, output_dim_order='XYZ'):
    """
    Given patches,  this function restores the original image.
    
    Parameters:
    patches (Numpy Array): Patches of size slice_indice x row_indice x 
    column_indice x image_slice x image_row x image_column.
    output_dim_order (String): Specifies the return dimension order. Could be
    'XYZ' or 'ZYX'
    
    Returns:
    slice_concat (Numpy Array): Original image out of the Concatenated patches.
    """
    
    # Extract the patches and build the volume
    for zslice in range(patches.shape[0]):
        for row in range(patches.shape[1]):
            for col in range(patches.shape[2]):
                # Extract a 3D-patch
                patch = patches[zslice, row, col, :]
                
                # First column-patch? -> Initialize a volume, else concatenate with the
                # last patch on the column-axis
                if(col == 0):
                    col_concat = patch
                else:
                    col_concat = np.concatenate((col_concat, patch), axis=2)
            
            # First row-patch? -> Initialize a volume, else concatenate with the
            # last patch on the row-axis
            if(row == 0):
                row_concat = col_concat
            else:
                row_concat = np.concatenate((row_concat, col_concat), axis=1)
        # First slice-patch? -> Initialize a volume, else concatenate with the
        # last patch on the slice-axis
        if(zslice == 0):
            slice_concat = row_concat
        else:
            slice_concat = np.concatenate((slice_concat, row_concat), axis=0)

    # The output volume is the over all three axes concatenated patches
    if output_dim_order == 'ZYX':
        return slice_concat
    if output_dim_order == 'XYZ':
        return np.transpose(slice_concat, axes=(2,1,0))

    