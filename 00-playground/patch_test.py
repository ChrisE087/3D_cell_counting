import nrrd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import sys
sys.path.append("..")
from tools import image_processing as impro

X_path = os.path.join('..', '..', '..', 'Daten2', 'Fibroblasten', '1_draq5.nrrd')
y_path = os.path.join('..', '..', '..', 'Daten2', 'Fibroblasten', '20190221_1547_Segmentation', '1_draq5-gauss_centroids.nrrd')

X, X_header = nrrd.read(X_path)
y, y_header = nrrd.read(y_path)

pz = py = px = 32
sz = sy = sx = 32

padding = 'SAME'


session = tf.Session()
p_X = impro.gen_patches(session=session, data=X, patch_slices=pz, patch_rows=py,
                        patch_cols=px, stride_slices=sz, stride_rows=sy,
                        stride_cols=sx, input_dim_order='XYZ', padding=padding)

p_y = impro.gen_patches(session=session, data=y, patch_slices=pz, patch_rows=py,
                        patch_cols=px, stride_slices=sz, stride_rows=sy,
                        stride_cols=sx, input_dim_order='XYZ', padding=padding)

border = None#(8,8,8)
X_r = impro.restore_volume(patches=p_X, border=border, output_dim_order='XYZ')
y_r = impro.restore_volume(patches=p_y, border=border, output_dim_order='XYZ')

plt.imshow(X_r[150,])
plt.imshow(y_r[150,])

nrrd.write('X_r.nrrd', X_r)
nrrd.write('y_r.nrrd', y_r)
