import numpy as np
from keras.models import Model, load_model, model_from_json
from keras.layers import Input, Conv3D, Conv3DTranspose, ReLU, LeakyReLU, Activation, MaxPooling3D, UpSampling3D, Cropping3D, BatchNormalization
from keras.layers.merge import concatenate
from keras import backend as K
from keras.regularizers import l2
import os
import sys
sys.path.append("..")
from tools import image_processing as impro

class CNN():
    
    def __init__(self, linear_output_scaling_factor, standardization_mode):
        
        print('Initializing NeuralNet')
        self.model = None
        self.linear_output_scaling_factor = linear_output_scaling_factor
        self.standardization_mode=standardization_mode
        
    def get_crop_shape(self, target, refer):
        
        # Nothing to do if the shapes are identical
        if target.get_shape().as_list() == refer.get_shape().as_list():
            print('nothing to crop')
            return(0, 0), (0, 0), (0, 0)
        else:
            # Calculate the crop factor of the depth (4th dimension)
            cd = np.abs(target.get_shape().as_list()[3] - refer.get_shape().as_list()[3])
            assert(cd >= 0)
            if cd % 2 != 0:
                cd1, cd2 = int(cd/2), int(cd/2) + 1
            else:
                cd1, cd2 = int(cd/2), int(cd/2)
            
            # Calculate the crop factor of the width (3th dimension)
            cw = np.abs(target.get_shape().as_list()[2] - refer.get_shape().as_list()[2])
            assert (cw >= 0)
            if cw % 2 != 0:
                cw1, cw2 = int(cw/2), int(cw/2) + 1
            else:
                cw1, cw2 = int(cw/2), int(cw/2)
            
            # Calculate the crop factor of the height (2th dimension)
            ch = np.abs(target.get_shape().as_list()[1] - refer.get_shape().as_list()[1])
            assert (ch >= 0)
            if ch % 2 != 0:
                ch1, ch2 = int(ch/2), int(ch/2) + 1
            else:
                ch1, ch2 = int(ch/2), int(ch/2)
        
            return (ch1, ch2), (cw1, cw2), (cd1, cd2)
        
###############################################################################
# First Net (works good)
###############################################################################
    
#    def define_model(self, input_shape, filters_exp=5, kernel_size=(3, 3, 3), 
#                  pool_size=(2, 2, 2), hidden_layer_activation='relu', 
#                  output_layer_activation=None, padding='same'):
#        
#        inputs = Input((input_shape))
#        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(inputs)
#        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(conv1)
#        pool1 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(conv1)
#    
#        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(pool1)
#        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(conv2)
#        pool2 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(conv2)
#    
#        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(pool2)
#        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(conv3)
#        pool3 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(conv3)
#    
#        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(pool3)
#        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(conv4)
#        pool4 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(conv4)
#    
#        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(pool4)
#        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(conv5)
#    
#        up_conv5 = UpSampling3D(size=pool_size)(conv5)
#        ch, cw, cd = self.get_crop_shape(up_conv5, conv4)
#        crop_conv4 = Cropping3D(cropping=(ch, cw, cd))(conv4)
#        up6 = concatenate([up_conv5, crop_conv4], axis=-1)
#        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(up6)
#        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(conv6)
#    
#        up_conv6 = UpSampling3D(size=pool_size)(conv6)
#        ch, cw, cd = self.get_crop_shape(up_conv6, conv3)
#        crop_conv3 = Cropping3D(cropping=(ch, cw, cd))(conv3)
#        up7 = concatenate([up_conv6, crop_conv3], axis=-1)
#        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(up7)
#        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(conv7)
#    
#        up_conv7 = UpSampling3D(size=pool_size)(conv7)
#        ch, cw, cd = self.get_crop_shape(up_conv7, conv2)
#        crop_conv2 = Cropping3D(cropping=(ch, cw, cd))(conv2)
#        up8 = concatenate([up_conv7, crop_conv2], axis=-1)
#        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(up8)
#        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(conv8)
#    
#        up_conv8 = UpSampling3D(size=pool_size)(conv8)
#        ch, cw, cd = self.get_crop_shape(up_conv8, conv1)
#        crop_conv1 = Cropping3D(cropping=(ch, cw, cd))(conv1)
#        up9 = concatenate([up_conv8, crop_conv1], axis=-1)
#        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(up9)
#        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(conv9)
#    
#        conv10 = Conv3D(filters=1, kernel_size=(1, 1, 1), 
#                        activation=output_layer_activation)(conv9)
#    
#        self.model = Model(inputs=[inputs], outputs=[conv10])
            
        
###############################################################################
# Another good one
###############################################################################
#    def define_model(self, input_shape, filters_exp=5, kernel_size=(3, 3, 3), 
#                  pool_size=(2, 2, 2), hidden_layer_activation='relu', 
#                  output_layer_activation=None, padding='same'):
#        norm_axis = -1
#        
#        inputs = Input((input_shape))
#        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(inputs)
#        conv1 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv1)
#        #conv1 = BatchNormalization(axis=norm_axis)(conv1)
#        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv1)
#        conv1 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv1)
#        conv1 = BatchNormalization(axis=norm_axis)(conv1)
#        pool1 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(conv1)
#    
#        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool1)
#        conv2 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv2)
#        #conv2 = BatchNormalization(axis=norm_axis)(conv2)
#        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv2)
#        conv2 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv2)
#        conv2 = BatchNormalization(axis=norm_axis)(conv2)
#        pool2 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(conv2)
#    
#        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool2)
#        conv3 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv3)
#        #conv3 = BatchNormalization(axis=norm_axis)(conv3)
#        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv3)
#        conv3 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv3)
#        conv3 = BatchNormalization(axis=norm_axis)(conv3)
#        pool3 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(conv3)
#    
#        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool3)
#        conv4 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv4)
#        #conv4 = BatchNormalization(axis=norm_axis)(conv4)
#        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv4)
#        conv4 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv4)
#        conv4 = BatchNormalization(axis=norm_axis)(conv4)
#        pool4 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(conv4)
#    
#        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool4)
#        conv5 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv5)
#        #conv5 = BatchNormalization(axis=norm_axis)(conv5)
#        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv5)
#        conv5 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv5)
#        conv5 = BatchNormalization(axis=norm_axis)(conv5)
#    
#        up_conv5 = UpSampling3D(size=pool_size)(conv5)
#        ch, cw, cd = self.get_crop_shape(up_conv5, conv4)
#        crop_conv4 = Cropping3D(cropping=(ch, cw, cd))(conv4)
#        up6 = concatenate([up_conv5, crop_conv4], axis=-1)
#        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up6)
#        conv6 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv6)
#        #conv6 = BatchNormalization(axis=norm_axis)(conv6)
#        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv6)
#        conv6 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv6)
#        #conv6 = BatchNormalization(axis=norm_axis)(conv6)
#    
#        up_conv6 = UpSampling3D(size=pool_size)(conv6)
#        ch, cw, cd = self.get_crop_shape(up_conv6, conv3)
#        crop_conv3 = Cropping3D(cropping=(ch, cw, cd))(conv3)
#        up7 = concatenate([up_conv6, crop_conv3], axis=-1)
#        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up7)
#        conv7 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv7)
#        #conv7 = BatchNormalization(axis=norm_axis)(conv7)
#        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv7)
#        conv7 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv7)
#        #conv7 = BatchNormalization(axis=norm_axis)(conv7)
#    
#        up_conv7 = UpSampling3D(size=pool_size)(conv7)
#        ch, cw, cd = self.get_crop_shape(up_conv7, conv2)
#        crop_conv2 = Cropping3D(cropping=(ch, cw, cd))(conv2)
#        up8 = concatenate([up_conv7, crop_conv2], axis=-1)
#        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up8)
#        conv8 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv8)
#        #conv8 = BatchNormalization(axis=norm_axis)(conv8)
#        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv8)
#        conv8 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv8)
#        #conv8 = BatchNormalization(axis=norm_axis)(conv8)
#    
#        up_conv8 = UpSampling3D(size=pool_size)(conv8)
#        ch, cw, cd = self.get_crop_shape(up_conv8, conv1)
#        crop_conv1 = Cropping3D(cropping=(ch, cw, cd))(conv1)
#        up9 = concatenate([up_conv8, crop_conv1], axis=-1)
#        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up9)
#        conv9 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv9)
#        #conv9 = BatchNormalization(axis=norm_axis)(conv9)
#        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv9)
#        conv9 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv9)
#        #conv9 = BatchNormalization(axis=norm_axis)(conv9)
#    
#        conv10 = Conv3D(filters=1, kernel_size=(1, 1, 1), 
#                        activation=output_layer_activation)(conv9)
#    
#        self.model = Model(inputs=[inputs], outputs=[conv10])
        
    
#        up_conv5 = UpSampling3D(size=pool_size)(conv5)
#        ch, cw, cd = self.get_crop_shape(up_conv5, conv4)
#        crop_conv4 = Cropping3D(cropping=(ch, cw, cd))(conv4)
#        up6 = concatenate([up_conv5, crop_conv4], axis=-1)
#        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(up6)
#        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(conv6)
#    
#        up_conv6 = UpSampling3D(size=pool_size)(conv6)
#        ch, cw, cd = self.get_crop_shape(up_conv6, conv3)
#        crop_conv3 = Cropping3D(cropping=(ch, cw, cd))(conv3)
#        up7 = concatenate([up_conv6, crop_conv3], axis=-1)
#        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(up7)
#        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(conv7)
#    
#        up_conv7 = UpSampling3D(size=pool_size)(conv7)
#        ch, cw, cd = self.get_crop_shape(up_conv7, conv2)
#        crop_conv2 = Cropping3D(cropping=(ch, cw, cd))(conv2)
#        up8 = concatenate([up_conv7, crop_conv2], axis=-1)
#        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(up8)
#        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(conv8)
#    
#        up_conv8 = UpSampling3D(size=pool_size)(conv8)
#        ch, cw, cd = self.get_crop_shape(up_conv8, conv1)
#        crop_conv1 = Cropping3D(cropping=(ch, cw, cd))(conv1)
#        up9 = concatenate([up_conv8, crop_conv1], axis=-1)
#        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(up9)
#        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(conv9)
#    
#        conv10 = Conv3D(filters=1, kernel_size=(1, 1, 1), 
#                        activation=output_layer_activation)(conv9)
#    
#        self.model = Model(inputs=[inputs], outputs=[conv10])



    def define_model(self, input_shape, filters_exp=5, kernel_size=(3, 3, 3), 
                      pool_size=(2, 2, 2), hidden_layer_activation='relu', 
                      output_layer_activation=None, regularization=None, 
                      padding='same'):
        norm_axis = -1
        if regularization != None:
            kernel_regularizer = l2(regularization)
        else:
            kernel_regularizer = None
        
        inputs = Input((input_shape))
        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
                       activation=None, padding=padding, 
                       kernel_regularizer=kernel_regularizer)(inputs)
        conv1 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv1)
        conv1 = BatchNormalization(axis=norm_axis)(conv1)
        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
                       activation=None, padding=padding,
                       kernel_regularizer=kernel_regularizer)(conv1)
        conv1 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv1)
        conv1 = BatchNormalization(axis=norm_axis)(conv1)
        pool1 = MaxPooling3D(pool_size=pool_size, strides=None, 
                             padding=padding)(conv1)
        
        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
                       activation=None, padding=padding,
                       kernel_regularizer=kernel_regularizer)(pool1)
        conv2 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv2)
        conv2 = BatchNormalization(axis=norm_axis)(conv2)
        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
                       activation=None, padding=padding,
                       kernel_regularizer=kernel_regularizer)(conv2)
        conv2 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv2)
        conv2 = BatchNormalization(axis=norm_axis)(conv2)
        pool2 = MaxPooling3D(pool_size=pool_size, strides=None, 
                             padding=padding)(conv2)
        
        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
                       activation=None, padding=padding,
                       kernel_regularizer=kernel_regularizer)(pool2)
        conv3 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv3)
        conv3 = BatchNormalization(axis=norm_axis)(conv3)
        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
                       activation=None, padding=padding,
                       kernel_regularizer=kernel_regularizer)(conv3)
        conv3 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv3)
        conv3 = BatchNormalization(axis=norm_axis)(conv3)
        pool3 = MaxPooling3D(pool_size=pool_size, strides=None, 
                             padding=padding)(conv3)
        
        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
                       activation=None, padding=padding,
                       kernel_regularizer=kernel_regularizer)(pool3)
        conv4 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv4)
        conv4 = BatchNormalization(axis=norm_axis)(conv4)
        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
                       activation=None, padding=padding,
                       kernel_regularizer=kernel_regularizer)(conv4)
        conv4 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv4)
        conv4 = BatchNormalization(axis=norm_axis)(conv4)
        pool4 = MaxPooling3D(pool_size=pool_size, strides=None, 
                             padding=padding)(conv4)
        
        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
                       activation=None, padding=padding,
                       kernel_regularizer=kernel_regularizer)(pool4)
        conv5 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv5)
        conv5 = BatchNormalization(axis=norm_axis)(conv5)
        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
                       activation=None, padding=padding,
                       kernel_regularizer=kernel_regularizer)(conv5)
        conv5 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv5)
        conv5 = BatchNormalization(axis=norm_axis)(conv5)
        
        up_conv5 = UpSampling3D(size=pool_size)(conv5)
        ch, cw, cd = self.get_crop_shape(up_conv5, conv4)
        crop_conv4 = Cropping3D(cropping=(ch, cw, cd))(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=-1)
        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
                       activation=None, padding=padding,
                       kernel_regularizer=kernel_regularizer)(up6)
        conv6 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv6)
        #conv6 = BatchNormalization(axis=norm_axis)(conv6)
        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
                       activation=None, padding=padding,
                       kernel_regularizer=kernel_regularizer)(conv6)
        conv6 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv6)
        #conv6 = BatchNormalization(axis=norm_axis)(conv6)
        
        up_conv6 = UpSampling3D(size=pool_size)(conv6)
        ch, cw, cd = self.get_crop_shape(up_conv6, conv3)
        crop_conv3 = Cropping3D(cropping=(ch, cw, cd))(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=-1)
        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
                       activation=None, padding=padding,
                       kernel_regularizer=kernel_regularizer)(up7)
        conv7 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv7)
        #conv7 = BatchNormalization(axis=norm_axis)(conv7)
        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
                       activation=None, padding=padding,
                       kernel_regularizer=kernel_regularizer)(conv7)
        conv7 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv7)
        #conv7 = BatchNormalization(axis=norm_axis)(conv7)
        
        up_conv7 = UpSampling3D(size=pool_size)(conv7)
        ch, cw, cd = self.get_crop_shape(up_conv7, conv2)
        crop_conv2 = Cropping3D(cropping=(ch, cw, cd))(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=-1)
        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
                       activation=None, padding=padding,
                       kernel_regularizer=kernel_regularizer)(up8)
        conv8 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv8)
        #conv8 = BatchNormalization(axis=norm_axis)(conv8)
        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
                       activation=None, padding=padding,
                       kernel_regularizer=kernel_regularizer)(conv8)
        conv8 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv8)
        #conv8 = BatchNormalization(axis=norm_axis)(conv8)
        
        up_conv8 = UpSampling3D(size=pool_size)(conv8)
        ch, cw, cd = self.get_crop_shape(up_conv8, conv1)
        crop_conv1 = Cropping3D(cropping=(ch, cw, cd))(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=-1)
        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
                       activation=None, padding=padding,
                       kernel_regularizer=kernel_regularizer)(up9)
        conv9 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv9)
        #conv9 = BatchNormalization(axis=norm_axis)(conv9)
        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
                       activation=None, padding=padding,
                       kernel_regularizer=kernel_regularizer)(conv9)
        conv9 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv9)
        #conv9 = BatchNormalization(axis=norm_axis)(conv9)
        
        conv10 = Conv3D(filters=1, kernel_size=(1, 1, 1), 
                        activation=output_layer_activation)(conv9)
        
        self.model = Model(inputs=[inputs], outputs=[conv10])







        
#    def define_model(self, input_shape, filters_exp=5, kernel_size=(3, 3, 3), 
#                  pool_size=(2, 2, 2), hidden_layer_activation='relu', 
#                  output_layer_activation=None, padding='same'):
#        norm_axis = 1
#        inputs = Input((input_shape))
#        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(inputs)
#        norm1 = BatchNormalization(axis=norm_axis)(conv1)
#        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(norm1)
#        norm1 = BatchNormalization(axis=norm_axis)(conv1)
#        pool1 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(norm1)
#    
#        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(pool1)
#        norm2 = BatchNormalization(axis=norm_axis)(conv2)
#        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(norm2)
#        norm2 = BatchNormalization(axis=norm_axis)(conv2)
#        pool2 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(norm2)
#    
#        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(pool2)
#        norm3 = BatchNormalization(axis=norm_axis)(conv3)
#        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(norm3)
#        norm3 = BatchNormalization(axis=norm_axis)(conv3)
#        pool3 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(norm3)
#    
#        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(pool3)
#        norm4 = BatchNormalization(axis=norm_axis)(conv4)
#        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(norm4)
#        norm4 = BatchNormalization(axis=norm_axis)(conv4)
#        pool4 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(norm4)
#    
#        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(pool4)
#        norm5 = BatchNormalization(axis=norm_axis)(conv5)
#        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(norm5)
#        norm5 = BatchNormalization(axis=norm_axis)(conv5)
#    
#        up_conv5 = UpSampling3D(size=pool_size)(norm5)
#        ch, cw, cd = self.get_crop_shape(up_conv5, norm4)
#        crop_conv4 = Cropping3D(cropping=(ch, cw, cd))(norm4)
#        up6 = concatenate([up_conv5, crop_conv4], axis=-1)
#        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(up6)
#        norm6 = BatchNormalization(axis=norm_axis)(conv6)
#        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(norm6)
#        norm6 = BatchNormalization(axis=norm_axis)(conv6)
#    
#        up_conv6 = UpSampling3D(size=pool_size)(norm6)
#        ch, cw, cd = self.get_crop_shape(up_conv6, norm3)
#        crop_conv3 = Cropping3D(cropping=(ch, cw, cd))(norm3)
#        up7 = concatenate([up_conv6, crop_conv3], axis=-1)
#        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(up7)
#        norm7 = BatchNormalization(axis=norm_axis)(conv7)
#        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(norm7)
#        norm7 = BatchNormalization(axis=norm_axis)(conv7)
#    
#        up_conv7 = UpSampling3D(size=pool_size)(norm7)
#        ch, cw, cd = self.get_crop_shape(up_conv7, norm2)
#        crop_conv2 = Cropping3D(cropping=(ch, cw, cd))(norm2)
#        up8 = concatenate([up_conv7, crop_conv2], axis=-1)
#        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(up8)
#        norm8 = BatchNormalization(axis=-2)(conv8)
#        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(norm8)
#        norm8 = BatchNormalization(axis=norm_axis)(conv8)
#    
#        up_conv8 = UpSampling3D(size=pool_size)(norm8)
#        ch, cw, cd = self.get_crop_shape(up_conv8, norm1)
#        crop_conv1 = Cropping3D(cropping=(ch, cw, cd))(norm1)
#        up9 = concatenate([up_conv8, crop_conv1], axis=-1)
#        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(up9)
#        norm9 = BatchNormalization(axis=norm_axis)(conv9)
#        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=hidden_layer_activation, padding=padding)(norm9)
#        norm9 = BatchNormalization(axis=norm_axis)(conv9)
#    
#        conv10 = Conv3D(filters=1, kernel_size=(1, 1, 1), 
#                        activation=output_layer_activation)(norm9)
#    
#        self.model = Model(inputs=[inputs], outputs=[conv10])
        
        
#    def conv3d_block(self, input_tensor, n_filters, kernel_size=3, padding='same', 
#                     activation='relu', batchnorm=True):
#        norm_axis = -1
#        # First Conv->Activation->Batch-Norm layer
#        x = Conv3D(filters=n_filters, kernel_size=(kernel_size, kernel_size, kernel_size), padding=padding,
#                   activation=None, use_bias=False, kernel_initializer='glorot_uniform')(input_tensor)
#        if activation == 'leaky_relu':
#            x = LeakyReLU(alpha=0.3)(x)
#        else:
#            x = Activation('relu')(x)
#        if batchnorm:
#            x = BatchNormalization(axis=norm_axis)(x)
#        # Second Conv->Activation->Batch-Norm layer
#        x = Conv3D(filters=n_filters, kernel_size=(kernel_size, kernel_size, kernel_size), padding=padding,
#                   activation=None, use_bias=False, kernel_initializer='glorot_uniform')(x)
#        if activation == 'leaky_relu':
#            x = LeakyReLU(alpha=0.3)(x)
#        else:
#            x = Activation('relu')(x)
#        if batchnorm:
#            x = BatchNormalization(axis=norm_axis)(x)
#        return x
#    
#    def define_unet(self, input_shape, n_filters=16, kernel_size=3, 
#                  batchnorm=True, hidden_layer_activation='relu',
#                  output_layer_activation=None, pool_size=2, padding='same'):
#        
#        inputs = Input((input_shape))
#        c1 = self.conv3d_block(inputs, n_filters=n_filters*1, kernel_size=kernel_size, 
#                          activation=hidden_layer_activation, batchnorm=batchnorm)
#        p1 = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size), strides=None, 
#                             padding=padding)(c1)
#        
#        c2 = self.conv3d_block(p1, n_filters=n_filters*2, kernel_size=kernel_size, 
#                          activation=hidden_layer_activation, batchnorm=batchnorm)
#        p2 = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size), strides=None, 
#                             padding=padding)(c2)
#        
#        c3 = self.conv3d_block(p2, n_filters=n_filters*4, kernel_size=kernel_size, 
#                          activation=hidden_layer_activation, batchnorm=batchnorm)
#        p3 = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size), strides=None, 
#                             padding=padding)(c3)
#        
#        c4 = self.conv3d_block(p3, n_filters=n_filters*8, kernel_size=kernel_size, 
#                          activation=hidden_layer_activation, batchnorm=batchnorm)
#        p4 = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size), strides=None, 
#                             padding=padding)(c4)
#        
#        c5 = self.conv3d_block(p4, n_filters=n_filters*16, kernel_size=kernel_size, 
#                          activation=hidden_layer_activation, batchnorm=batchnorm)
#        
#        u6 = UpSampling3D(size=(pool_size, pool_size, pool_size))(c5)
#        u6 = concatenate([u6, c4], axis=-1)
#        c6 = self.conv3d_block(u6, n_filters=n_filters*8, kernel_size=kernel_size, 
#                          activation=hidden_layer_activation, batchnorm=batchnorm)
#        
#        u7 = UpSampling3D(size=(pool_size, pool_size, pool_size))(c6)
#        u7 = concatenate([u7, c3], axis=-1)
#        c7 = self.conv3d_block(u7, n_filters=n_filters*4, kernel_size=kernel_size, 
#                          activation=hidden_layer_activation, batchnorm=batchnorm)
#        
#        u8 = UpSampling3D(size=(pool_size, pool_size, pool_size))(c7)
#        u8 = concatenate([u8, c2], axis=-1)
#        c8 = self.conv3d_block(u8, n_filters=n_filters*2, kernel_size=kernel_size, 
#                          activation=hidden_layer_activation, batchnorm=batchnorm)
#        
#        u9 = UpSampling3D(size=(pool_size, pool_size, pool_size))(c8)
#        u9 = concatenate([u9, c1], axis=-1)
#        c9 = self.conv3d_block(u9, n_filters=n_filters*1, kernel_size=kernel_size, 
#                          activation=hidden_layer_activation, batchnorm=batchnorm)
#        
#        outputs = Conv3D(filters=1, kernel_size=(1, 1, 1), 
#                        activation=output_layer_activation)(c9)
#        
#        self.model = Model(inputs=[inputs], outputs=[outputs])
        

        
    def compile_model(self, loss, optimizer, metrics):
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, 
                           loss_weights=None, sample_weight_mode=None, 
                           weighted_metrics=None, target_tensors=None)
        return self.model.summary()
        
    def fit(self, X, y, batch_size, epochs, callbacks, validation_split, 
            validation_data, shuffle=True):
        
        history = self.model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, 
                                 verbose=1, callbacks=callbacks, validation_split=validation_split, 
                                 validation_data=validation_data, shuffle=True)
        return history
    
    def fit_generator(self, epochs, train_generator, val_generator, callbacks):
        
        history = self.model.fit_generator(generator=train_generator, 
                                 steps_per_epoch=None, epochs=epochs, 
                                 verbose=1, callbacks=callbacks, 
                                 validation_data=val_generator, 
                                 validation_steps=None, class_weight=None, 
                                 max_queue_size=10, workers=1, 
                                 use_multiprocessing=False, 
                                 shuffle=True, initial_epoch=0)
        return history
        
    def evaluate_model(self, X_test, y_test, batch_size):

        # Standardize the model input
        if self.standardization_mode == 'per_slice' or \
        self.standardization_mode == 'per_sample' or \
        self.standardization_mode == 'per_batch':
            print('Standardizing test data with mode: ', self.standardization_mode)
            X_test = impro.standardize_data(data=X_test, mode=self.standardization_mode)
        else:
            print('Test data is not standardized.')
            
        test_loss = self.model.evaluate(x=X_test, y=y_test, batch_size=batch_size, verbose=1, 
                            sample_weight=None, steps=None)
        return test_loss
        
    def predict_sample(self, X_pred):
        
        # Standardize the model input
        if self.standardization_mode == 'per_slice' or \
        self.standardization_mode == 'per_sample' or \
        self.standardization_mode == 'per_batch':
            #print('Standardizing input data with mode: ', self.standardization_mode)
            #print('Standardization function: impro.standardize_data')
            X_pred = impro.standardize_data(data=X_pred, mode=self.standardization_mode)
            #print('Standardization function: impro.standardize_3d_image')
            #X_pred = impro.standardize_3d_image(X_pred)

        # Expand the dims for batches and channels
        X_pred = np.expand_dims(np.expand_dims(X_pred, axis=3), axis=0)

        # Predict y
        y_pred = self.model.predict(X_pred)
        
        # Undo the linear scaling
        y_pred = y_pred[0,:,:,:,0]/self.linear_output_scaling_factor
        
        return y_pred
    
    def predict_batch(self, X_pred_batch):
        
        # Standardize the model input
        if self.standardization_mode == 'per_slice' or \
        self.standardization_mode == 'per_sample' or \
        self.standardization_mode == 'per_batch':
            #print('Standardizing input data with mode: ', self.standardization_mode)
            X_pred_batch = impro.standardize_data(data=X_pred_batch, mode=self.standardization_mode)
        
        # Expand the dims for channels
        X_pred_batch = np.expand_dims(X_pred_batch, axis=4)
        
        # Predict y
        y_pred_batch = self.model.predict(X_pred_batch)
        
        # Undo the linear scaling
        y_pred_batch = y_pred_batch[0,:,:,:,0]/self.linear_output_scaling_factor
        
        return y_pred_batch
    
    def save_model_single_file(self, path, model_name):
        export_path = os.path.join(path, model_name+'.hdf5')
        self.model.save(export_path)
        
    def save_model_json(self, path, model_name):
        export_path = os.path.join(path, model_name+'.json')
        model_json = self.model.to_json()
        with open(export_path, 'w') as json_file:
            json_file.write(model_json)
        json_file.close()
        
    def save_model_weights(self, path, weights_name):
        export_path = os.path.join(path, weights_name+'.hdf5')
        self.model.save_weights(export_path)
        
    def load_model_single_file(self, path, model_name):
        import_path = os.path.join(path, model_name+'.hdf5')
        self.model = load_model(import_path)
        
    def load_model_json(self, path, model_name, weights_name):
        json_import_path = os.path.join(path, model_name+'.json')
        weights_import_path = os.path.join(path, weights_name+'.hdf5')
        json_file = open(json_import_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(weights_import_path)
        
    
        
        
        