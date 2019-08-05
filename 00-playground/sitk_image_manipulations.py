import numpy as np
import nrrd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

#%% Load image
data_path = os.path.join('..', '..', '..', 'Daten', 'extracted_cells', '09.nrrd')
cell_np, header = nrrd.read(data_path)
cell_np = np.transpose(cell_np, axes=(2,1,0))
plt.imshow(cell_np[10,])

#%% Make a SimpleITK image
cell_itk = sitk.GetImageFromArray(cell_np)
spacings = header.get('space directions')
spacings = np.array([spacings[0,0], spacings[1,1], spacings[2,2]])
cell_itk.SetSpacing(spacings)
print(cell_itk.GetSpacing())

#%% Gaussian Filter
gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
gaussian.SetSigma (5.0)
cell_itk_gauss = gaussian.Execute(cell_itk)

pixelID = cell_itk_gauss.GetPixelID()
caster = sitk.CastImageFilter()
caster.SetOutputPixelType( pixelID )
cell_itk_gauss = caster.Execute( cell_itk_gauss )


cell_np_gauss = sitk.GetArrayFromImage(cell_itk_gauss)
plt.imshow(cell_np[10,])
