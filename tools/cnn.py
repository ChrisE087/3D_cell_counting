import numpy as np
from keras.models import Model
from keras.layers import Input, Conv3D, Conv3DTranspose, LeakyReLU, MaxPooling3D, UpSampling3D, Cropping3D
from keras.layers.merge import concatenate
from keras import backend as K
import os
import sys
sys.path.append("..")
from tools import image_processing as impro

class CNN():
    
    def __init__(self):
        
        print('Initializing NeuralNet')
        self.model = None
        
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
    
    def define_model(self, input_shape, filters_exp=5, kernel_size=(3, 3, 3), 
                  pool_size=(2, 2, 2), hidden_layer_activation='relu', 
                  output_layer_activation=None, padding='same'):
        
        inputs = Input((input_shape))
        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
                       activation=hidden_layer_activation, padding=padding)(inputs)
        conv1 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
                       activation=hidden_layer_activation, padding=padding)(conv1)
        pool1 = MaxPooling3D(pool_size=pool_size, strides=None, 
                             padding=padding)(conv1)
    
        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
                       activation=hidden_layer_activation, padding=padding)(pool1)
        conv2 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
                       activation=hidden_layer_activation, padding=padding)(conv2)
        pool2 = MaxPooling3D(pool_size=pool_size, strides=None, 
                             padding=padding)(conv2)
    
        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
                       activation=hidden_layer_activation, padding=padding)(pool2)
        conv3 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
                       activation=hidden_layer_activation, padding=padding)(conv3)
        pool3 = MaxPooling3D(pool_size=pool_size, strides=None, 
                             padding=padding)(conv3)
    
        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
                       activation=hidden_layer_activation, padding=padding)(pool3)
        conv4 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
                       activation=hidden_layer_activation, padding=padding)(conv4)
        pool4 = MaxPooling3D(pool_size=pool_size, strides=None, 
                             padding=padding)(conv4)
    
        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
                       activation=hidden_layer_activation, padding=padding)(pool4)
        conv5 = Conv3D(filters=2**filters_exp+4, kernel_size=kernel_size, 
                       activation=hidden_layer_activation, padding=padding)(conv5)
    
        up_conv5 = UpSampling3D(size=pool_size)(conv5)
        ch, cw, cd = self.get_crop_shape(up_conv5, conv4)
        crop_conv4 = Cropping3D(cropping=(ch, cw, cd))(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=-1)
        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
                       activation=hidden_layer_activation, padding=padding)(up6)
        conv6 = Conv3D(filters=2**filters_exp+3, kernel_size=kernel_size, 
                       activation=hidden_layer_activation, padding=padding)(conv6)
    
        up_conv6 = UpSampling3D(size=pool_size)(conv6)
        ch, cw, cd = self.get_crop_shape(up_conv6, conv3)
        crop_conv3 = Cropping3D(cropping=(ch, cw, cd))(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=-1)
        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
                       activation=hidden_layer_activation, padding=padding)(up7)
        conv7 = Conv3D(filters=2**filters_exp+2, kernel_size=kernel_size, 
                       activation=hidden_layer_activation, padding=padding)(conv7)
    
        up_conv7 = UpSampling3D(size=pool_size)(conv7)
        ch, cw, cd = self.get_crop_shape(up_conv7, conv2)
        crop_conv2 = Cropping3D(cropping=(ch, cw, cd))(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=-1)
        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
                       activation=hidden_layer_activation, padding=padding)(up8)
        conv8 = Conv3D(filters=2**filters_exp+1, kernel_size=kernel_size, 
                       activation=hidden_layer_activation, padding=padding)(conv8)
    
        up_conv8 = UpSampling3D(size=pool_size)(conv8)
        ch, cw, cd = self.get_crop_shape(up_conv8, conv1)
        crop_conv1 = Cropping3D(cropping=(ch, cw, cd))(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=-1)
        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
                       activation=hidden_layer_activation, padding=padding)(up9)
        conv9 = Conv3D(filters=2**filters_exp, kernel_size=kernel_size, 
                       activation=hidden_layer_activation, padding=padding)(conv9)
    
        conv10 = Conv3D(filters=1, kernel_size=(1, 1, 1), 
                        activation=output_layer_activation)(conv9)
    
        self.model = Model(inputs=[inputs], outputs=[conv10])
        
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
        export_path = os.path.join(path, model_name+'.h5')
        self.model.save(export_path)
        
    def save_model_json(self, path, model_name):
        export_path = os.path.join(path, model_name+'.json')
        model_json = self.model.to_json
        with open(export_path, 'w') as json_file:
            json_file.write(model_json)
        json_file.close()
        
    def save_model_weights(self, path, weights_name):
        export_path = os.path.join(path, weights_name+'.h5')
        self.model.save_weights(export_path)
        
        
        