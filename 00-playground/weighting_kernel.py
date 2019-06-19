import numpy as np

def get_weight_matrix(patch_size, strides):
    patch_size_z = patch_size[0]
    patch_size_y = patch_size[1]
    patch_size_x = patch_size[2]
    
    stride_z = strides[0]
    stride_y = strides[1]
    stride_x = strides[2]
    
    weight_matrix = np.ones(shape=(patch_size_z, patch_size_y, patch_size_x))
    
    weight_matrix[0:stride_z, stride_y:, 0:stride_x] *= 1/2
    weight_matrix[0:stride_z, 0:stride_y, stride_x:] *= 1/2
    weight_matrix[stride_z:, 0:stride_y:, 0:stride_x] *= 1/2
    weight_matrix[0:stride_z, stride_y:, stride_x:] *= 1/4
    weight_matrix[stride_z:, 0:stride_y, stride_x:] *= 1/4
    weight_matrix[stride_z:, stride_y:, 0:stride_x] *= 1/4
    weight_matrix[stride_z:, stride_y:, stride_x:] *= 1/8
    
    return weight_matrix

patch_size_z = 32
patch_size_y = 32
patch_size_x = 32

stride_z = 24
stride_y = 24
stride_x = 24

overlap_z = patch_size_z - stride_z
overlap_y = patch_size_y - stride_y
overlap_x = patch_size_x - stride_x

weight_matrix = get_weight_matrix((8,8,8), (6,6,6))               
