import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nrrd
import os

#img_path = os.path.join('test_data', 'test_object.nrrd')
img_path = os.path.join('test_data', 'test_object_with_physical_voxel_size.nrrd')
img, metadata = nrrd.read(img_path) # XYZ
img = img / 255
img = np.transpose(img, (2, 1, 0)) # ZYX
img = sitk.GetImageFromArray(img)
space_directions = metadata.get('space directions')
img.SetSpacing((space_directions[0,0], space_directions[1,1], space_directions[2,2]))

cc = sitk.ConnectedComponent(img>0)

statistics = sitk.LabelIntensityStatisticsImageFilter()
statistics.Execute(cc, img)

# Make a white dot [f(x,y,z)=1] at each centroid of the label l
for l in statistics.GetLabels():
    phy_size = statistics.GetPhysicalSize(l)
    print(phy_size)
