import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Conv3D, LeakyReLU, MaxPooling3D, UpSampling3D


batch_size = 16
input_shape = (32, 32, 32, 1)

# Create the model
inputs = Input(shape=input_shape)
conv_1 = Conv3D(16, (3, 3, 3), strides=(1, 1, 1), padding='valid', activation='relu') (inputs)
conv_2 = Conv3D(16, (3, 3, 3), strides=(1, 1, 1), padding='valid', activation='relu') (conv_1)
act_1 = LeakyReLU(alpha=0.2)(conv_2)
pool_1 = MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid')(act_1)
conv_3 = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='valid', activation='relu')(pool_1)
conv_4 = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='valid', activation='relu')(conv_3)
act_2 = LeakyReLU(alpha=0.2)(conv_4)
pool_2 = MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid')(act_2)
conv_5 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='valid', activation='relu')(pool_2)
conv_6 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='valid', activation='relu')(conv_5)
act_3 = LeakyReLU(alpha=0.2)(conv_6)
up_1 = UpSampling3D(size=(2, 2, 2))(act_3)
conv_7 = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='valid', activation='relu')(up_1)
conv_8 = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='valid', activation='relu')(conv_7)
act_4 = LeakyReLU(alpha=0.2)(conv_8)
up_2 = UpSampling3D(size=(2, 2, 2))(act_4)
conv_9 = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='valid', activation='relu')(up_2)
conv_10 = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='valid', activation='relu')(conv_9)
act_4 = LeakyReLU(alpha=0.2)(conv_10)
out = Conv3D(1, (1, 1, 1), activation='sigmoid')(act_4)
model = Model(inputs, out)
model.summary()