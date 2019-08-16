import nrrd
import numpy as np
import os 
import SimpleITK as sitk
import cc3d

def filter_segmentation(segmentation, spacings, excluded_volume_size):
    """
    This method sets segmented cells in a 3d numpy array with volume-sizes 
    in (um^3) < ecluded_volume_size to zero.
    
    Parameters:
    segmentation (Numpy Array): 3D Numpy array of 3D cell segmentations with 
    dimension order XYZ
    spacings (Numpy Array): Array of physical Voxel size in dimension order XYZ
    excluded_volumes (float): Volume size in um^3. Segmentations for cell volumes
    smaller than this value are set to zero
    
    Returns:
    filtered_segmentation (Numpy Array): 3D Numpy array with the same dimension
    as the input parameter in which all segmentations < excluded_volume_size
    are set to zero.
    """
    # Transpose to ZYX for use with SimpleITK and make a SimpleITK image
    segmentation = np.transpose(segmentation, axes=(2,1,0)) # ZYX
    seg_itk = sitk.GetImageFromArray(segmentation, isVector=False) # XYZ
    seg_itk.SetSpacing(spacings)

    # Get The Connected Components of the volume image. All intensities greater 
    # than 0 are taken into account for the labeling
    cc = sitk.ConnectedComponent(seg_itk>0)
    
    # Calculate the statitics of the labeled regions
    statistics = sitk.LabelIntensityStatisticsImageFilter()
    statistics.Execute(cc, seg_itk)
    
    # Make a new volume for the result
    seg_filtered = np.copy(segmentation) # ZYX
    
    for l in statistics.GetLabels():
        # Calculate the physical volume size for the segmentation with the label l
        phy_size = statistics.GetPhysicalSize(l)
        
        # Notice only volumes smaller than a specific volume size, all others set to zero
        if phy_size < excluded_volume_size:
            
            # If the volume is smaller than the threshold, zero the segmentation
            seg_filtered[seg_filtered == l] = 0
        
    # Make a binary segmentation out of the labelled segmentation
    seg_filtered[seg_filtered > 0] = 1
    
    # Assing new labels to the binary segmentation
    seg_filtered = cc3d.connected_components(seg_filtered, connectivity=6)
    
    # Transpose back to XYZ for saving to disk
    segmentation = np.transpose(segmentation, axes=(2,1,0)) # XYZ
    seg_filtered = np.transpose(seg_filtered, axes=(2,1,0)) # XYZ
    
    return seg_filtered
    

#path_to_seg = os.path.join('..', '..', '..', 'Daten2', 'Fibroblasten', '20190221_1547_Segmentation', '5_draq5-NucleiBinary.nrrd')
path_to_seg = os.path.join('..', '..', '..', 'Daten', '24h', 'untreated', 'C1-untreated_1.1_OpenSegSPIMResults_', 'Nucleisegmentedfill2r_labelled.nrrd')
seg, header = nrrd.read(path_to_seg) #XYZ

#spacings = header.get('spacings')
space_directions = header.get('space directions')
spacings = np.array((space_directions[0,0],space_directions[1,1], space_directions[2,2]))

excluded_volume_size = 3 #um^3

print(np.max(seg))
seg_filtered = filter_segmentation(seg, spacings, excluded_volume_size)


print(np.max(seg_filtered))
nrrd.write('seg_filtered.nrrd', seg_filtered, header=header)
