from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

#todo: modify loss function...


# dice-coef-loss
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet_3d(vol_x, vol_y, vol_z, chn, optimizer=Adam(lr=0.00001), loss=dice_coef_loss,
             metrics=[dice_coef]):

    print('Data format: ' + K.image_data_format())

    if K.image_data_format() == 'channels_first':
        input_dims = (chn, vol_x, vol_y, vol_z)
        feat_axis = 1
    else:
        input_dims = (vol_x, vol_y, vol_z, chn)
        feat_axis = 4

    # u-net model
    inputs = Input(shape=input_dims)
    conv1 = Conv3D(32, (3, 3, 3), activation=None, padding='same')(inputs) #Conv3D --> (filters, kernel_size, ...)
    conv1 = BatchNormalization(axis=feat_axis, scale=False)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv3D(64, (3, 3, 3), activation=None, padding='same')(conv1)
    conv1 = BatchNormalization(axis=feat_axis, scale=False)(conv1)
    conv1 = Activation('relu')(conv1)

    pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation=None, padding='same')(pool1)
    conv2 = BatchNormalization(axis=feat_axis, scale=False)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv3D(128, (3, 3, 3), activation=None, padding='same')(conv2)
    conv2 = BatchNormalization(axis=feat_axis, scale=False)(conv2)
    conv2 = Activation('relu')(conv2)

    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation=None, padding='same')(pool2)
    conv3 = BatchNormalization(axis=feat_axis, scale=False)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv3D(256, (3, 3, 3), activation=None, padding='same')(conv3)
    conv3 = BatchNormalization(axis=feat_axis, scale=False)(conv3)
    conv3 = Activation('relu')(conv3)

    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation=None, padding='same')(pool3)
    conv4 = BatchNormalization(axis=feat_axis, scale=False)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv3D(512, (3, 3, 3), activation=None, padding='same')(conv4)
    conv4 = BatchNormalization(axis=feat_axis, scale=False)(conv4)
    conv4 = Activation('relu')(conv4)

    up1 = UpSampling3D(size=(2, 2, 2))(conv4)
    up1 = Concatenate(axis=feat_axis)([conv3, up1])

    upconv1 = Conv3D(256, (3, 3, 3), activation=None, padding='same')(up1)
    upconv1 = BatchNormalization(axis=feat_axis, scale=False)(upconv1)
    upconv1 = Activation('relu')(upconv1)
    upconv1 = Conv3D(256, (3, 3, 3), activation=None, padding='same')(upconv1)
    upconv1 = BatchNormalization(axis=feat_axis, scale=False)(upconv1)
    upconv1 = Activation('relu')(upconv1)

    up2 = UpSampling3D(size=(2, 2, 2))(upconv1)
    up2 = Concatenate(axis=feat_axis)([conv2, up2])

    upconv2 = Conv3D(128, (3, 3, 3), activation=None, padding='same')(up2)
    upconv2 = BatchNormalization(axis=feat_axis, scale=False)(upconv2)
    upconv2 = Activation('relu')(upconv2)
    upconv2 = Conv3D(128, (3, 3, 3), activation=None, padding='same')(upconv2)
    upconv2 = BatchNormalization(axis=feat_axis, scale=False)(upconv2)
    upconv2 = Activation('relu')(upconv2)

    up3 = UpSampling3D(size=(2, 2, 2))(upconv2)
    up3 = Concatenate(axis=feat_axis)([conv1, up3])

    upconv3 = Conv3D(64, (3, 3, 3), activation=None, padding='same')(up3)
    upconv3 = BatchNormalization(axis=feat_axis, scale=False)(upconv3)
    upconv3 = Activation('relu')(upconv3)
    upconv3 = Conv3D(64, (3, 3, 3), activation=None, padding='same')(upconv3)
    upconv3 = BatchNormalization(axis=feat_axis, scale=False)(upconv3)
    upconv3 = Activation('relu')(upconv3)

    conv_final = Conv3D(4, (3, 3, 3), activation='sigmoid', padding='same')(upconv3)

    model = Model(inputs=inputs, outputs=conv_final)
    model.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

def get_unet_2d(sz_x, sz_y, chn, optimizer=Adam(lr=0.00001), loss=dice_coef_loss,
             metrics=[dice_coef]):

    print('Data format: ' + K.image_data_format())

    if K.image_data_format() == 'channels_first':
        input_dims = (chn, sz_x, sz_y)
        feat_axis = 1
    else:
        input_dims = (sz_x, sz_y, chn)
        feat_axis = 3

    # u-net model
    inputs = Input(shape=input_dims)
    conv1 = Conv2D(32, (3, 3), activation=None, padding='same')(inputs)
    conv1 = BatchNormalization(axis=feat_axis, scale=False)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3, 3), activation=None, padding='same')(conv1)
    conv1 = BatchNormalization(axis=feat_axis, scale=False)(conv1)
    conv1 = Activation('relu')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation=None, padding='same')(pool1)
    conv2 = BatchNormalization(axis=feat_axis, scale=False)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2(128, (3, 3), activation=None, padding='same')(conv2)
    conv2 = BatchNormalization(axis=feat_axis, scale=False)(conv2)
    conv2 = Activation('relu')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation=None, padding='same')(pool2)
    conv3 = BatchNormalization(axis=feat_axis, scale=False)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, (3, 3), activation=None, padding='same')(conv3)
    conv3 = BatchNormalization(axis=feat_axis, scale=False)(conv3)
    conv3 = Activation('relu')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation=None, padding='same')(pool3)
    conv4 = BatchNormalization(axis=feat_axis, scale=False)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, (3, 3), activation=None, padding='same')(conv4)
    conv4 = BatchNormalization(axis=feat_axis, scale=False)(conv4)
    conv4 = Activation('relu')(conv4)

    up1 = UpSampling2D(size=(2, 2))(conv4)
    up1 = Concatenate(axis=feat_axis)([conv3, up1])

    upconv1 = Conv2D(256, (3, 3), activation=None, padding='same')(up1)
    upconv1 = BatchNormalization(axis=feat_axis, scale=False)(upconv1)
    upconv1 = Activation('relu')(upconv1)
    upconv1 = Conv2D(256, (3, 3), activation=None, padding='same')(upconv1)
    upconv1 = BatchNormalization(axis=feat_axis, scale=False)(upconv1)
    upconv1 = Activation('relu')(upconv1)

    up2 = UpSampling2D(size=(2, 2))(upconv1)
    up2 = Concatenate(axis=feat_axis)([conv2, up2])

    upconv2 = Conv2D(128, (3, 3), activation=None, padding='same')(up2)
    upconv2 = BatchNormalization(axis=feat_axis, scale=False)(upconv2)
    upconv2 = Activation('relu')(upconv2)
    upconv2 = Conv2D(128, (3, 3), activation=None, padding='same')(upconv2)
    upconv2 = BatchNormalization(axis=feat_axis, scale=False)(upconv2)
    upconv2 = Activation('relu')(upconv2)

    up3 = UpSampling2D(size=(2, 2))(upconv2)
    up3 = Concatenate(axis=feat_axis)([conv1, up3])

    upconv3 = Conv2D(64, (3, 3), activation=None, padding='same')(up3)
    upconv3 = BatchNormalization(axis=feat_axis, scale=False)(upconv3)
    upconv3 = Activation('relu')(upconv3)
    upconv3 = Conv2D(64, (3, 3), activation=None, padding='same')(upconv3)
    upconv3 = BatchNormalization(axis=feat_axis, scale=False)(upconv3)
    upconv3 = Activation('relu')(upconv3)

    conv_final = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(upconv3)

    model = Model(inputs=inputs, outputs=conv_final)
    model.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
