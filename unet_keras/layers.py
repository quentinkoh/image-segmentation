# import the necessary packages
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from keras.optimizers import RMSprop


def unet_encode(inputs, num_filters):
    '''
    Encoding path of the U-Net network
    ---
    Args:
        inputs: tensor by the shape of (width, height, channels)
        num_filters: number of convolving filters

    Returns:
        x
    '''
    x = Conv2D(num_filters, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def unet_maxpool(inputs):
    '''
    MaxPooling layer for encoding path of the U-Net network
    ---
    Args:
        inputs: tensor by the shape of (width, height, channels)

    Returns:
        x
    '''
    x = MaxPooling2D((2, 2), strides=(2, 2))(inputs)

    return x


def unet_decode(up_input, down_input, num_filters):
    '''
    Decoding path of the U-Net network
    ---
    Args:
        up_input: tensor from previous layer
        down_input: tensor from encoding layer
        num_filters: number of convolving filters

    Returns:
        x
    '''
    x = UpSampling2D((2, 2))(up_input)
    x = concatenate([down_input, x], axis=3)
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x
