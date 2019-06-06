import numpy as np
import matplotlib.pyplot as plt
import nrrd
import keras
from keras.models import Model
from keras.layers import Input, Conv3D, Conv3DTranspose, LeakyReLU, MaxPooling3D, UpSampling3D, Cropping3D
from keras.layers.merge import concatenate
from keras import backend as K

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_crop_shape(target, refer):
    #print('Refer: ', refer.get_shape())
    #print('Target: ', target.get_shape())
    
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

def get_model(input_shape, filters_exp, activation, padding):
    inputs = Input((input_shape))
    conv1 = Conv3D(filters=2**filters_exp, kernel_size=(3, 3, 3), activation=activation, padding=padding)(inputs)
    conv1 = Conv3D(filters=2**filters_exp, kernel_size=(3, 3, 3), activation=activation, padding=padding)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding=padding)(conv1)

    conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=(3, 3, 3), activation=activation, padding=padding)(pool1)
    conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=(3, 3, 3), activation=activation, padding=padding)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding=padding)(conv2)

    conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=(3, 3, 3), activation=activation, padding=padding)(pool2)
    conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=(3, 3, 3), activation=activation, padding=padding)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding=padding)(conv3)

    conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=(3, 3, 3), activation=activation, padding=padding)(pool3)
    conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=(3, 3, 3), activation=activation, padding=padding)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding=padding)(conv4)

    conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=(3, 3, 3), activation=activation, padding=padding)(pool4)
    conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=(3, 3, 3), activation=activation, padding=padding)(conv5)

    up_conv5 = UpSampling3D(size=(2, 2, 2))(conv5)
    ch, cw, cd = get_crop_shape(up_conv5, conv4)
    crop_conv4 = Cropping3D(cropping=(ch, cw, cd))(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=-1)
    conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=(3, 3, 3), activation=activation, padding=padding)(up6)
    conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=(3, 3, 3), activation=activation, padding=padding)(conv6)

    up_conv6 = UpSampling3D(size=(2, 2, 2))(conv6)
    ch, cw, cd = get_crop_shape(up_conv6, conv3)
    crop_conv3 = Cropping3D(cropping=(ch, cw, cd))(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=-1)
    conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=(3, 3, 3), activation=activation, padding=padding)(up7)
    conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=(3, 3, 3), activation=activation, padding=padding)(conv7)

    up_conv7 = UpSampling3D(size=(2, 2, 2))(conv7)
    ch, cw, cd = get_crop_shape(up_conv7, conv2)
    crop_conv2 = Cropping3D(cropping=(ch, cw, cd))(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=-1)
    conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=(3, 3, 3), activation=activation, padding=padding)(up8)
    conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=(3, 3, 3), activation=activation, padding=padding)(conv8)

    up_conv8 = UpSampling3D(size=(2, 2, 2))(conv8)
    ch, cw, cd = get_crop_shape(up_conv8, conv1)
    crop_conv1 = Cropping3D(cropping=(ch, cw, cd))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=-1)
    conv9 = Conv3D(filters=2**filters_exp, kernel_size=(3, 3, 3), activation=activation, padding=padding)(up9)
    conv9 = Conv3D(filters=2**filters_exp, kernel_size=(3, 3, 3), activation=activation, padding=padding)(conv9)

    conv10 = Conv3D(filters=1, kernel_size=(1, 1, 1), activation='linear')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model

# Load the dataset
X_train, X_train_metadata = nrrd.read('test_data/X.nrrd') # Shape: number, XYZ
y_train, y_train_metadata = nrrd.read('test_data/y.nrrd') # Shape: number, XYZ

# Expand the dimensions -> Channel = 1, because images are grayscale
X_train = np.expand_dims(X_train, axis=4)
y_train = np.expand_dims(y_train, axis=4)

# Normalize the dataset in range 0-1
X_train = X_train/255
y_train = y_train/65535

model = get_model(X_train.shape[1:], 5, 'relu', 'same')

opt = keras.optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae', 'accuracy'])
model.fit(X_train, y_train, batch_size=16, epochs=32)

num = 568

to_predict = np.expand_dims(X_train[num,:,:,:], axis=0)
y_truth = y_train[num,:,:,:]
plt.imshow(y_truth[:,:,5,0])

plt.imshow(to_predict[0,:,:,5,0])


y_pred = model.predict(to_predict)
plt.imshow(y_pred[0,:,:,8,0])

print(np.sum(y_truth))
print(np.sum(y_pred))
