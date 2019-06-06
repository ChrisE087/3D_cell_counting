import numpy as np
import matplotlib.pyplot as plt
import nrrd
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Conv3D, LeakyReLU, MaxPooling3D, UpSampling3D
from keras import backend as K

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# Load the dataset
X_train, X_train_metadata = nrrd.read('test_data/X.nrrd') # Shape: number, XYZ
y_train, y_train_metadata = nrrd.read('test_data/y.nrrd') # Shape: number, XYZ

# Expand the dimensions -> Channel = 1, because images are grayscale
X_train = np.expand_dims(X_train, axis=4)
y_train = np.expand_dims(y_train, axis=4)

# Normalize the dataset in range 0-1
X_train = X_train/255
y_train = y_train/65535

# Build the model
model = Sequential()
model.add(Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last', activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last', activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='same'))
model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last', activation='relu'))
model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last', activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='same'))
model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last', activation='relu'))
model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last', activation='relu'))
model.add(UpSampling3D(size=(2, 2, 2)))
model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last', activation='relu'))
model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last', activation='relu'))
model.add(UpSampling3D(size=(2, 2, 2)))
model.add(Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last', activation='relu'))
model.add(Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_last', activation='relu'))
model.add(Conv3D(1, (1, 1, 1), activation='sigmoid'))

opt = keras.optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss=dice_coef_loss, optimizer=opt, metrics=[dice_coef])

model.fit(X_train, y_train, batch_size=8, epochs=64)

num = 68

to_predict = np.expand_dims(X_train[num,:,:,:], axis=0)
y_truth = y_train[num,:,:,:]

plt.imshow(to_predict[0,:,:,4,0])
plt.imshow(y_truth[:,:,4,0])

y_pred = model.predict(to_predict)
plt.imshow(y_pred[0,:,:,7,0])

print(np.sum(y_truth))
print(np.sum(y_pred))
