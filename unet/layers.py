# ================ Layer Constructors ================ #
# import the necessary packages
import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


def conv(inputs, kernel_size, num_outputs, name,
         stride_size=[1, 1], padding='SAME', activation_fn=tf.nn.relu):
    '''
    Convolutional layer followed by ReLU Activation:
    ---
    Args:
           inputs: tensor by the shape of (batch_size, height, width, channels)
           kernel_size: size of convolving filter by (height, width) in list
           num_outputs: number of convolving filters in integer
           name: scope name for tf.variable_scope() in string
           stride_size: convolution stride by (heigh, width) in list
           padding: input padding in string
           activation_fn: activation function on output (can be None)
    Returns:
               outputs: tensor by shape of (batch_size, height, width, num_outputs)
    '''
    with tf.variable_scope(name):
        num_filters_in = input.get_shape()[-1].value
        kernel_shape = [kernel_size[0], kernel_size[1], num_filters_in, num_outputs]
        stride_shape = [1, stride_size[0], stride_size[1], 1]

        weights = tf.get_variable(name='weights', shape=kernel_shape, dtype=tf.float32,
                                  initalizer=xavier_initializer())
        bias = tf.get_variable(name='bias', shape=[num_outputs], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input=inputs, filter=weights,
                            strides=stride_shape, padding=padding)
        outputs = tf.nn.bias_add(value=conv, bias=bias)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

            return outputs


def conv_bn(inputs, kernel_size, num_outputs, name,
            is_training=True, stride_size=[1, 1], padding='SAME', activation_fn=tf.nn.relu):
    '''
        Convolutional layer followed by Batch Normalization then ReLU Activation:
        ---
        Args:
                inputs: tensor by the shape of (batch_size, height, width, channels)
                kernel_size: size of convolving filter by (height, width) in list
                num_outputs: number of convolving filters in integer
                name: scope name for tf.variable_scope() in string
                is_training: in training mode or not in boolean
                stride_size: convolution stride by (height, width) in list
                padding: input padding in string
                activation_fn: activation function on output (can be None)

        Returns:
                outputs: tensor by shape of (batch_size, height, width, num_outputs)
        '''

    with tf.variable_scope(name):
        num_filters_in = inputs.get_shape()[-1].value
        kernel_shape = [kernel_size[0], kernel_size[1], num_filters_in, num_outputs]
        stride_shape = [1, stride_size[0], stride_size[1], 1]

        weights = tf.get_variable(name='weights', shape=kernel_shape, dtype=tf.float32,
                                  initializer=xavier_initializer())
        bias = tf.get_variable(name='bias', shape=[num_outputs], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input=inputs, filter=weights, strides=stride_shape,
                            padding=padding)
        outputs = tf.nn.bias_add(value=conv, bias=bias)

        # tf.layers.batch_normalization can be used as well
        outputs = tf.contrib.layers.batch_norm(outputs, center=True, scale=True,
                                               is_training=is_training)

        if activation_fn is not None:
            outputs = activation_fn(ouputs)

        return outputs


def maxpooling_layer(inputs, kernel_size, name, padding='SAME'):
    '''
    MaxPooling layer (also known as downsampling layer):
    ---
    Args:
            inputs: tensor by the shape of (batch_size, height, width, channels)
            kernel_size: size of convolving filter by (height, width) in list
            name: scope name for tf.variable_scope() in string
            padding: input padding in string

    Returns:
            outputs: tensor by shape of
            (batch_size, height / kernel_size[0], width / kernel_size[1], channels)
    '''
    kernel_shape = [1, kernel_size[0], kernel_size[1], 1]
    outputs = tf.nn.max_pool(value=inputs, ksize=kernel_shape, strides=kernel_shape,
                             padding=padding, name=name)

    return outputs


def dropout_layer(inputs, keep_prob, name):
    '''
    Dropout layer (drops out a random set of activations in this layer by setting them to zero)
                (regularizer to prevent over-fitting)
    ---
    Args:
        inputs: tensor by the shape of (batch_size, height, width, channels)
        keep_prob: probability of keeping each element in the tensor
        name: scope name for tf.variable_scope() in string

    Returns:
        outputs: tensor by the shape of (batch_size, height, width, channels)
    '''

    return tf.nn.dropout(x=inputs, keep_prob=keep_prob, name=name)


def upsample_layer(inputs, factor, name, padding='SAME', activation_fn=None):
    '''
    Convolution tranpose upsampling layer with bilinear interpolation weights
    Using bilinear interpolation to upsample the spatial resolution of input volume
    # issue: problems with odd scaling factors
    ---
    Args:
        inputs: tensor by the shape of (batch_size, heigth, width, channels)
        factor: upsampling factor in integer
        name: scope name for tf.variable_scope() in string
        padding: input padding in string
        activation_fn: activation function on output (can be None)

    Returns:
        outputs: tensor by the shape of (batch_size, height * factor, width, channels)
    '''
    with tf.variable_scope(name):
        stride_shape = [1, factor, factor, 1]
        input_shape = tf.shape(input=inputs)
        num_filters_in = inputs.get_shape()[-1].value
        output_shape = tf.stack([input_shape[0], input_shape[1] * factor,
                                 input_shape[2] * factor, num_filters_in])

        weights = bilinear_weights(factor, num_filters_in)
        outputs = tf.nn.conv2d_transpose(
            input=inputs, filter=weights, strides=stride_shape, padding=padding)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def bilinear_weights(factor, num_outputs):
    '''
    Create weights matrix for tranposed convolution with bilinear interpolation
    ---
    Args:
        factor: upsampling factor for the transposed convolution in integer
        num_outputs: number of convolving filters in integer

    Returns:
        weights: tensor by the shape of (kernel_size, kernel_size, num_outputs)
    '''
    kernel_size = 2 * factor - factor % 2

    weights_kernel = np.zeros((kernel_size, kernel_size, num_outputs, num_outputs,
                               num_outputs), dtype=np.float32)

    r_factor = (kernel_sze + 1) // 2
    if kernel_size % 2 == 1:
        center = r_factor - 1
    else:
        center = r_factor - 0.5

    grid = np.ogrid[:kernel_size, :kernel_size]
    upsample_kernel = (1 - abs(grid[0] - center) / r_factor) * \
        (1 - abs(grid[1] - center) / r_factor)

    for i in xrange(num_outputs):
        weights_kernel[:, :, i, i] = upsample_kernel

    init = tf.constant_initializer(value=weights_kernel, dtype=tf.float32)
    weights = tf.get_variable(name='weights', shape=weights_kernel.shape, dtype=tf.float32,
                              initializer=init)

    return weights


def concat(inputs1, inputs2, name):
    '''
    Concatenate two tensors
    ---
    Args:
        inputs1: tensor by the shape of (batch_size, height, width, channels)
        inputs2: tensor by the shape of (batch_size, height, width, channels)
        name: scope name for tf.variable_scope() in string

    Returns:
        outputs: tensor by the shape of (batch_size, height, width, channels1 + channels2)
    '''

    return tf.concat(values=[inputs1, inputs2], axis=3, name=name)
