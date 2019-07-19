import sys
sys.path.append("..")
import os
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

#path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'Daten', '24h', 'untreated'))
path_to_data = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', '..', 'Daten'))
interpolator = 'bspline'

for directory in os.listdir(path_to_data):
    data_dir = os.path.join(path_to_data, directory, 'untreated')
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.tif'):
                path_to_tif = os.path.join(data_dir, filename)
                print('Processing image: ', path_to_tif)
                
                # Get a numpy array from the tif stack with the dimension
                meta_data, raw_data = bfio.get_tif_stack(filepath=path_to_tif, series=0, depth='z', return_dim_order='XYZC') # XYZC
                
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
                #print(image.GetOrigin())
                #print(image.GetDirection())
                #print(image.GetSpacing())
                
                # Make isotropic voxels. Distinction needed, so that 
                # 48h->untreated_3 and 72h->untreated_1 have the same z-size as
                # the corresponding OpenSegSPIM-data
                if ('48h' in path_to_tif) and ('untreated_3' in path_to_tif):
                    # Here the rounding of th calculation is right, add 0 to the z-size
                    resampled_image = impro.make_image_isotropic(image, interpolator, 0)
                elif ('72h' in path_to_tif) and ('untreated_1' in path_to_tif):
                    # Here the rounding of th calculation is right, add 0 to the z-size
                    resampled_image = impro.make_image_isotropic(image, interpolator, 0)
                else:
                    # Here the rounding of th calculation is wrong, minus 1 to the z-size
                    resampled_image = impro.make_image_isotropic(image, interpolator, -1)
                
                #print(resampled_image.GetOrigin())
                #print(resampled_image.GetDirection())
                #print(resampled_image.GetSpacing())
                
                # Get a numpy array from the resampled simpleITK image
                np_image = sitk.GetArrayFromImage(resampled_image)
                
                # Transpose the numpy array from ZYX back to to XYZ
                np_image = np.transpose(np_image, axes=[2,1,0]) # XYZ
                np_image = np_image.astype('uint8')
                
                
                
                new_spacing = resampled_image.GetSpacing()
                header = {"spacings": [new_spacing[0], new_spacing[1], new_spacing[2]], 
                          "dimension": np_image.ndim,
                          "type": "uchar", 
                          "sizes": [resampled_image.GetWidth(), resampled_image.GetHeight(), 
                                    resampled_image.GetDepth()],
                          "units": ['"microns"', '"microns"', '"microns"']}
                name = os.path.splitext(filename)[0]
                new_filename = os.path.join(data_dir, name)
                new_filename = new_filename+'_8_bit'+'.nrrd'
                nrrd.write(new_filename, data=np_image, header=header, index_order='F')


    
        
