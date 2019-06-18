import numpy as np
from keras.models import Model, load_model, model_from_json
from keras.layers import Input, Conv3D, Conv3DTranspose, ReLU, LeakyReLU, MaxPooling3D, UpSampling3D, Cropping3D, BatchNormalization
from keras.layers.merge import concatenate
from keras import backend as K
import os
import sys
sys.path.append("..")
from tools import image_processing as impro

class CNN():
    
    def __init__(self, linear_output_scaling_factor):
        
        print('Initializing NeuralNet')
        self.model = None
        self.linear_output_scaling_factor = linear_output_scaling_factor
        
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
# Second Net with Batch Normalization Layers between Conv and Activation
###############################################################################

#    def define_model(self, input_shape, filters_exp=5, kernel_size=(3, 3, 3), 
#                  pool_size=(2, 2, 2), hidden_layer_activation='relu', 
#                  output_layer_activation=None, padding='same'):
#        if hidden_layer_activation == 'relu':
#            activation = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
#        if hidden_layer_activation == 'leaky_relu':
#            activation = LeakyReLU(alpha=0.3)
#        
#        inputs = Input((input_shape))
#        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(inputs)
#        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv1)
#        norm1 = BatchNormalization(axis=-2)(conv1)
#        act1 = activation(norm1)
#        pool1 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(act1)
#    
#        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool1)
#        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv2)
#        norm2 = BatchNormalization(axis=-2)(conv2)
#        act2 = activation(norm2)
#        pool2 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(act2)
#    
#        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool2)
#        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv3)
#        norm3 = BatchNormalization(axis=-2)(conv3)
#        act3 = activation(norm3)
#        pool3 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(act3)
#    
#        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool3)
#        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv4)
#        norm4 = BatchNormalization(axis=-2)(conv4)
#        act4 = activation(norm4)
#        pool4 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(act4)
#    
#        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool4)
#        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv5)
#        norm5 = BatchNormalization(axis=-2)(conv5)
#        act5 = activation(norm5)
#    
#        up_conv5 = UpSampling3D(size=pool_size)(act5)
#        ch, cw, cd = self.get_crop_shape(up_conv5, conv4)
#        crop_conv4 = Cropping3D(cropping=(ch, cw, cd))(conv4)
#        up6 = concatenate([up_conv5, crop_conv4], axis=-1)
#        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up6)
#        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv6)
#        norm6 = BatchNormalization(axis=-2)(conv6)
#        act6 = activation(norm6)
#        
#    
#        up_conv6 = UpSampling3D(size=pool_size)(act6)
#        ch, cw, cd = self.get_crop_shape(up_conv6, conv3)
#        crop_conv3 = Cropping3D(cropping=(ch, cw, cd))(conv3)
#        up7 = concatenate([up_conv6, crop_conv3], axis=-1)
#        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up7)
#        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv7)
#        norm7 = BatchNormalization(axis=-2)(conv7)
#        act7 = activation(norm7)
#    
#        up_conv7 = UpSampling3D(size=pool_size)(act7)
#        ch, cw, cd = self.get_crop_shape(up_conv7, conv2)
#        crop_conv2 = Cropping3D(cropping=(ch, cw, cd))(conv2)
#        up8 = concatenate([up_conv7, crop_conv2], axis=-1)
#        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up8)
#        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv8)
#        norm8 = BatchNormalization(axis=-2)(conv8)
#        act8 = activation(norm8)
#    
#        up_conv8 = UpSampling3D(size=pool_size)(act8)
#        ch, cw, cd = self.get_crop_shape(up_conv8, conv1)
#        crop_conv1 = Cropping3D(cropping=(ch, cw, cd))(conv1)
#        up9 = concatenate([up_conv8, crop_conv1], axis=-1)
#        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up9)
#        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv9)
#        norm9 = BatchNormalization(axis=-2)(conv9)
#        act9 = activation(norm9)
#    
#        conv10 = Conv3D(filters=1, kernel_size=(1, 1, 1), 
#                        activation=output_layer_activation)(act9)
#    
#        self.model = Model(inputs=[inputs], outputs=[conv10])


###############################################################################
# Third Net with Batch Normalization Layers after Activation
###############################################################################
      
#    def define_model(self, input_shape, filters_exp=5, kernel_size=(3, 3, 3), 
#                  pool_size=(2, 2, 2), hidden_layer_activation='relu', 
#                  output_layer_activation=None, padding='same'):
#        if hidden_layer_activation == 'relu':
#            activation = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
#        if hidden_layer_activation == 'leaky_relu':
#            activation = LeakyReLU(alpha=0.3)
#        
#        inputs = Input((input_shape))
#        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(inputs)
#        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv1)
#        act1 = activation(conv1)
#        norm1 = BatchNormalization(axis=-2)(act1)
#        pool1 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(norm1)
#    
#        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool1)
#        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv2)
#        act2 = activation(conv2)
#        norm2 = BatchNormalization(axis=-2)(act2)
#        pool2 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(norm2)
#    
#        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool2)
#        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv3)
#        act3 = activation(conv3)
#        norm3 = BatchNormalization(axis=-2)(act3)
#        pool3 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(norm3)
#    
#        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool3)
#        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv4)
#        act4 = activation(conv4)
#        norm4 = BatchNormalization(axis=-2)(act4)
#        pool4 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(norm4)
#    
#        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool4)
#        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv5)
#        act5 = activation(conv5)
#        norm5 = BatchNormalization(axis=-2)(act5)
#    
#        up_conv5 = UpSampling3D(size=pool_size)(norm5)
#        ch, cw, cd = self.get_crop_shape(up_conv5, conv4)
#        crop_conv4 = Cropping3D(cropping=(ch, cw, cd))(conv4)
#        up6 = concatenate([up_conv5, crop_conv4], axis=-1)
#        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up6)
#        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv6)
#        act6 = activation(conv6)
#        norm6 = BatchNormalization(axis=-2)(act6)
#        
#        up_conv6 = UpSampling3D(size=pool_size)(norm6)
#        ch, cw, cd = self.get_crop_shape(up_conv6, conv3)
#        crop_conv3 = Cropping3D(cropping=(ch, cw, cd))(conv3)
#        up7 = concatenate([up_conv6, crop_conv3], axis=-1)
#        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up7)
#        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv7)
#        act7 = activation(conv7)
#        norm7 = BatchNormalization(axis=-2)(act7)
#    
#        up_conv7 = UpSampling3D(size=pool_size)(norm7)
#        ch, cw, cd = self.get_crop_shape(up_conv7, conv2)
#        crop_conv2 = Cropping3D(cropping=(ch, cw, cd))(conv2)
#        up8 = concatenate([up_conv7, crop_conv2], axis=-1)
#        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up8)
#        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv8)
#        act8 = activation(conv8)
#        norm8 = BatchNormalization(axis=-2)(act8)
#    
#        up_conv8 = UpSampling3D(size=pool_size)(norm8)
#        ch, cw, cd = self.get_crop_shape(up_conv8, conv1)
#        crop_conv1 = Cropping3D(cropping=(ch, cw, cd))(conv1)
#        up9 = concatenate([up_conv8, crop_conv1], axis=-1)
#        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up9)
#        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(conv9)
#        act9 = activation(conv9)
#        norm9 = BatchNormalization(axis=-2)(act9)
#    
#        conv10 = Conv3D(filters=1, kernel_size=(1, 1, 1), 
#                        activation=output_layer_activation)(norm9)
#    
#        self.model = Model(inputs=[inputs], outputs=[conv10])
        
###############################################################################
# Third Net with additional Activation- and Normalization layers
###############################################################################
      
#    def define_model(self, input_shape, filters_exp=5, kernel_size=(3, 3, 3), 
#                  pool_size=(2, 2, 2), hidden_layer_activation='relu', 
#                  output_layer_activation=None, padding='same'):
#        if hidden_layer_activation == 'relu':
#            activation = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
#        if hidden_layer_activation == 'leaky_relu':
#            activation = LeakyReLU(alpha=0.2)
#        
#        inputs = Input((input_shape))
#        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(inputs)
#        act1 = activation(conv1)
#        norm1 = BatchNormalization(axis=-2)(act1)
#        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(norm1)
#        act1 = activation(conv1)
#        norm1 = BatchNormalization(axis=-2)(act1)
#        pool1 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(norm1)
#    
#        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool1)
#        act2 = activation(conv2)
#        norm2 = BatchNormalization(axis=-2)(act2)
#        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(norm2)
#        act2 = activation(conv2)
#        norm2 = BatchNormalization(axis=-2)(act2)
#        pool2 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(norm2)
#
#        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool2)
#        act3 = activation(conv3)
#        norm3 = BatchNormalization(axis=-2)(act3)
#        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(norm3)
#        act3 = activation(conv3)
#        norm3 = BatchNormalization(axis=-2)(act3)
#        pool3 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(norm3)
#    
#        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool3)
#        act4 = activation(conv4)
#        norm4 = BatchNormalization(axis=-2)(act4)
#        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(norm4)
#        act4 = activation(conv4)
#        norm4 = BatchNormalization(axis=-2)(act4)
#        pool4 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(norm4)
#    
#        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool4)
#        act5 = activation(conv5)
#        norm5 = BatchNormalization(axis=-2)(act5)
#        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(norm5)
#        act5 = activation(conv5)
#        norm5 = BatchNormalization(axis=-2)(act5)
#    
#        up_conv5 = UpSampling3D(size=pool_size)(norm5)
#        ch, cw, cd = self.get_crop_shape(up_conv5, conv4)
#        crop_conv4 = Cropping3D(cropping=(ch, cw, cd))(conv4)
#        up6 = concatenate([up_conv5, crop_conv4], axis=-1)
#        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up6)
#        act6 = activation(conv6)
#        norm6 = BatchNormalization(axis=-2)(act6)
#        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(norm6)
#        act6 = activation(conv6)
#        norm6 = BatchNormalization(axis=-2)(act6)
#        
#        up_conv6 = UpSampling3D(size=pool_size)(norm6)
#        ch, cw, cd = self.get_crop_shape(up_conv6, conv3)
#        crop_conv3 = Cropping3D(cropping=(ch, cw, cd))(conv3)
#        up7 = concatenate([up_conv6, crop_conv3], axis=-1)
#        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up7)
#        act7 = activation(conv7)
#        norm7 = BatchNormalization(axis=-2)(act7)
#        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(norm7)
#        act7 = activation(conv7)
#        norm7 = BatchNormalization(axis=-2)(act7)
#    
#        up_conv7 = UpSampling3D(size=pool_size)(norm7)
#        ch, cw, cd = self.get_crop_shape(up_conv7, conv2)
#        crop_conv2 = Cropping3D(cropping=(ch, cw, cd))(conv2)
#        up8 = concatenate([up_conv7, crop_conv2], axis=-1)
#        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up8)
#        act8 = activation(conv8)
#        norm8 = BatchNormalization(axis=-2)(act8)
#        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(norm8)
#        act8 = activation(conv8)
#        norm8 = BatchNormalization(axis=-2)(act8)
#    
#        up_conv8 = UpSampling3D(size=pool_size)(norm8)
#        ch, cw, cd = self.get_crop_shape(up_conv8, conv1)
#        crop_conv1 = Cropping3D(cropping=(ch, cw, cd))(conv1)
#        up9 = concatenate([up_conv8, crop_conv1], axis=-1)
#        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up9)
#        act9 = activation(conv9)
#        norm9 = BatchNormalization(axis=-2)(act9)
#        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(norm9)
#        act9 = activation(conv9)
#        norm9 = BatchNormalization(axis=-2)(act9)
#    
#        conv10 = Conv3D(filters=1, kernel_size=(1, 1, 1), 
#                        activation=output_layer_activation)(norm9)
#    
#        self.model = Model(inputs=[inputs], outputs=[conv10])
            
###############################################################################
# Fifth Net (same as Net four but with removed Normalization Layers  if no 
# Pooling is following)
###############################################################################
      
#    def define_model(self, input_shape, filters_exp=5, kernel_size=(3, 3, 3), 
#                  pool_size=(2, 2, 2), hidden_layer_activation='relu', 
#                  output_layer_activation=None, padding='same'):
#        if hidden_layer_activation == 'relu':
#            activation = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
#        if hidden_layer_activation == 'leaky_relu':
#            activation = LeakyReLU(alpha=0.2)
#        
#        inputs = Input((input_shape))
#        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(inputs)
#        act1 = activation(conv1)
#        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(act1)
#        act1 = activation(conv1)
#        norm1 = BatchNormalization(axis=-2)(act1)
#        pool1 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(norm1)
#    
#        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool1)
#        act2 = activation(conv2)
#        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(act2)
#        act2 = activation(conv2)
#        norm2 = BatchNormalization(axis=-2)(act2)
#        pool2 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(norm2)
#
#        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool2)
#        act3 = activation(conv3)
#        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(act3)
#        act3 = activation(conv3)
#        norm3 = BatchNormalization(axis=-2)(act3)
#        pool3 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(norm3)
#    
#        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool3)
#        act4 = activation(conv4)
#        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(act4)
#        act4 = activation(conv4)
#        norm4 = BatchNormalization(axis=-2)(act4)
#        pool4 = MaxPooling3D(pool_size=pool_size, strides=None, 
#                             padding=padding)(norm4)
#    
#        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(pool4)
#        act5 = activation(conv5)
#        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(act5)
#        act5 = activation(conv5)
#        norm5 = BatchNormalization(axis=-2)(act5)
#    
#        up_conv5 = UpSampling3D(size=pool_size)(norm5)
#        ch, cw, cd = self.get_crop_shape(up_conv5, conv4)
#        crop_conv4 = Cropping3D(cropping=(ch, cw, cd))(conv4)
#        up6 = concatenate([up_conv5, crop_conv4], axis=-1)
#        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up6)
#        act6 = activation(conv6)
#        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(act6)
#        act6 = activation(conv6)
#        norm6 = BatchNormalization(axis=-2)(act6)
#        
#        up_conv6 = UpSampling3D(size=pool_size)(norm6)
#        ch, cw, cd = self.get_crop_shape(up_conv6, conv3)
#        crop_conv3 = Cropping3D(cropping=(ch, cw, cd))(conv3)
#        up7 = concatenate([up_conv6, crop_conv3], axis=-1)
#        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up7)
#        act7 = activation(conv7)
#        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(act7)
#        act7 = activation(conv7)
#        norm7 = BatchNormalization(axis=-2)(act7)
#    
#        up_conv7 = UpSampling3D(size=pool_size)(norm7)
#        ch, cw, cd = self.get_crop_shape(up_conv7, conv2)
#        crop_conv2 = Cropping3D(cropping=(ch, cw, cd))(conv2)
#        up8 = concatenate([up_conv7, crop_conv2], axis=-1)
#        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up8)
#        act8 = activation(conv8)
#        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(act8)
#        act8 = activation(conv8)
#        norm8 = BatchNormalization(axis=-2)(act8)
#    
#        up_conv8 = UpSampling3D(size=pool_size)(norm8)
#        ch, cw, cd = self.get_crop_shape(up_conv8, conv1)
#        crop_conv1 = Cropping3D(cropping=(ch, cw, cd))(conv1)
#        up9 = concatenate([up_conv8, crop_conv1], axis=-1)
#        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(up9)
#        act9 = activation(conv9)
#        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
#                       activation=None, padding=padding)(act9)
#        act9 = activation(conv9)
#        #norm9 = BatchNormalization(axis=-2)(act9)
#    
#        conv10 = Conv3D(filters=1, kernel_size=(1, 1, 1), 
#                        activation=output_layer_activation)(act9)
#    
#        self.model = Model(inputs=[inputs], outputs=[conv10])
        

###############################################################################
# Sixth Net (same as Net five but with additional outputs for an additional 
# loss
###############################################################################
      
    def define_model(self, input_shape, filters_exp=5, kernel_size=(3, 3, 3), 
                  pool_size=(2, 2, 2), hidden_layer_activation='relu', 
                  output_layer_activation=None, padding='same'):
        if hidden_layer_activation == 'relu':
            activation = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
        if hidden_layer_activation == 'leaky_relu':
            activation = LeakyReLU(alpha=0.2)
        
        inputs = Input((input_shape))
        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
                       activation=None, padding=padding)(inputs)
        act1 = activation(conv1)
        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
                       activation=None, padding=padding)(act1)
        act1 = activation(conv1)
        norm1 = BatchNormalization(axis=-2)(act1)
        pool1 = MaxPooling3D(pool_size=pool_size, strides=None, 
                             padding=padding)(norm1)
    
        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
                       activation=None, padding=padding)(pool1)
        act2 = activation(conv2)
        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
                       activation=None, padding=padding)(act2)
        act2 = activation(conv2)
        norm2 = BatchNormalization(axis=-2)(act2)
        pool2 = MaxPooling3D(pool_size=pool_size, strides=None, 
                             padding=padding)(norm2)

        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
                       activation=None, padding=padding)(pool2)
        act3 = activation(conv3)
        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
                       activation=None, padding=padding)(act3)
        act3 = activation(conv3)
        norm3 = BatchNormalization(axis=-2)(act3)
        pool3 = MaxPooling3D(pool_size=pool_size, strides=None, 
                             padding=padding)(norm3)
    
        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
                       activation=None, padding=padding)(pool3)
        act4 = activation(conv4)
        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
                       activation=None, padding=padding)(act4)
        act4 = activation(conv4)
        norm4 = BatchNormalization(axis=-2)(act4)
        pool4 = MaxPooling3D(pool_size=pool_size, strides=None, 
                             padding=padding)(norm4)
    
        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
                       activation=None, padding=padding)(pool4)
        act5 = activation(conv5)
        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
                       activation=None, padding=padding)(act5)
        act5 = activation(conv5)
        norm5 = BatchNormalization(axis=-2)(act5)
        # Additional output
        conv5_out = Conv3D(filters=1, kernel_size=(1, 1, 1), 
                        activation=output_layer_activation)(act5)
        print('Shape Conv5_out: ', conv5_out.shape)
    
        up_conv5 = UpSampling3D(size=pool_size)(norm5)
        ch, cw, cd = self.get_crop_shape(up_conv5, conv4)
        crop_conv4 = Cropping3D(cropping=(ch, cw, cd))(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=-1)
        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
                       activation=None, padding=padding)(up6)
        act6 = activation(conv6)
        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
                       activation=None, padding=padding)(act6)
        act6 = activation(conv6)
        norm6 = BatchNormalization(axis=-2)(act6)
        # Additional output
        conv6_out = Conv3D(filters=1, kernel_size=(1, 1, 1), 
                        activation=output_layer_activation)(act6)
        print('Shape Conv6_out: ', conv6_out.shape)
        
        up_conv6 = UpSampling3D(size=pool_size)(norm6)
        ch, cw, cd = self.get_crop_shape(up_conv6, conv3)
        crop_conv3 = Cropping3D(cropping=(ch, cw, cd))(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=-1)
        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
                       activation=None, padding=padding)(up7)
        act7 = activation(conv7)
        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
                       activation=None, padding=padding)(act7)
        act7 = activation(conv7)
        norm7 = BatchNormalization(axis=-2)(act7)
        # Additional output
        conv7_out = Conv3D(filters=1, kernel_size=(1, 1, 1), 
                        activation=output_layer_activation)(act7)
        print('Shape Conv7_out: ', conv7_out.shape)
    
        up_conv7 = UpSampling3D(size=pool_size)(norm7)
        ch, cw, cd = self.get_crop_shape(up_conv7, conv2)
        crop_conv2 = Cropping3D(cropping=(ch, cw, cd))(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=-1)
        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
                       activation=None, padding=padding)(up8)
        act8 = activation(conv8)
        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
                       activation=None, padding=padding)(act8)
        act8 = activation(conv8)
        norm8 = BatchNormalization(axis=-2)(act8)
        # Additional output
        conv8_out = Conv3D(filters=1, kernel_size=(1, 1, 1), 
                        activation=output_layer_activation)(act8)
        print('Shape Conv8_out: ', conv8_out.shape)
    
        up_conv8 = UpSampling3D(size=pool_size)(norm8)
        ch, cw, cd = self.get_crop_shape(up_conv8, conv1)
        crop_conv1 = Cropping3D(cropping=(ch, cw, cd))(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=-1)
        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
                       activation=None, padding=padding)(up9)
        act9 = activation(conv9)
        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
                       activation=None, padding=padding)(act9)
        act9 = activation(conv9)
        #norm9 = BatchNormalization(axis=-2)(act9)
    
        main_output = Conv3D(filters=1, kernel_size=(1, 1, 1), 
                        activation=output_layer_activation)(act9)
    
        self.model = Model(inputs=[inputs], outputs=[main_output, conv5_out])
        
    def compile_model(self, loss, optimizer, metrics):
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, 
                           loss_weights=None, sample_weight_mode=None, 
                           weighted_metrics=None, target_tensors=None)
        
    def fit(self, X, y, batch_size, epochs, callbacks, validation_split, 
            validation_data, shuffle=True):
        
        history = self.model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, 
                                 verbose=1, callbacks=callbacks, validation_split=validation_split, 
                                 validation_data=validation_data, shuffle=True)
        return history
    
    def fit_generator(self, epochs, train_generator, val_generator, callbacks):
        
        history = self.model.fit_generator(generator=train_generator, 
                                 steps_per_epoch=None, epochs = epochs, 
                                 verbose=1, callbacks=callbacks, 
                                 validation_data=val_generator, 
                                 validation_steps=None, class_weight=None, 
                                 max_queue_size=10, workers=1, 
                                 use_multiprocessing=False, 
                                 shuffle=True, initial_epoch=0)
        return history
        
    def evaluate_model(self, X_test, y_test, batch_size=32):
        
        test_loss = self.model.evaluate(x=X_test, y=y_test, batch_size=batch_size, verbose=1, 
                            sample_weight=None, steps=None)
        return test_loss
        
    def predict_sample(self, X_pred):
        
        # Standardize the model input
        X_pred, mean, sigma = impro.standardize_data(X_pred)

        # Expand the dims for batches and channels
        X_pred = np.expand_dims(np.expand_dims(X_pred, axis=3), axis=0)

        # Predict y
        y_pred = self.model.predict(X_pred)
        y_pred = y_pred[0,:,:,:,0]/self.linear_output_scaling_factor
        
        return y_pred
    
    def predict_batch(self, X_pred_batch):
        
        # Standardize each model input in the batch
        for i in range(np.size(X_pred_batch[0])):
            X_pred = X_pred_batch[i,]
            X_pred, mean, sigma = impro.standardize_data(X_pred)
            X_pred_batch[i,] = X_pred
            
        # Expand the dims for channels
        X_pred_batch = np.expand_dims(X_pred_batch, axis=3)
        
        # Predict y
        y_pred_batch = self.model.predict(X_pred_batch)
        
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
        
    
        
        
        