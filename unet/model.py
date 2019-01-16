# import the necessary packages
import tensorflow as tf
import unet.layers as layers


def build(inputs, num_classes, is_training):
    dropout_keep_prob = tf.where(is_training, 0.2, 1.0)
    '''
    Build U-Net Network:
    ---
    Args:
        inputs: tensor by the shape of (batch_size, height, width, channels)
        num_classes: number of segmentation labels integer
        is_training: in training mode or not in boolean (for bn & dropout layers)

    Returns:
        logits: tensor (predicted flattened image)
                by the shape of (batch_size * height * width, num_classes)
    '''

    # ================ U-Net Encoder Section (contracting path) ================ #
    # with colored input volume (i.e. depth = 3)
    # note that .conv_bn() method under layers.py consists of the following operation: Conv => BN => ReLU

    # Block 1
    conv11 = layers.conv_bn(inputs=inputs, kernel_size=[
                            3, 3], num_outputs=64, name='conv11', is_training=is_training)
    conv12 = layers.conv_bn(inputs=conv11, kernel_size=[
                            3, 3], num_outputs=64, name='conv12', is_training=is_training)
    pool1 = layers.maxpooling_layer(inputs=conv12, kernel_size=[2, 2], name='pool1')

    # Block 2
    conv21 = layers.conv_bn(inputs=pool1, kernel_size=[
                            3, 3], num_outputs=128, name='conv21', is_training=is_training)
    conv22 = layers.conv_bn(inputs=conv21, kernel_size=[
                            3, 3], num_outputs=128, name='conv22', is_training=is_training)
    pool2 = layers.maxpooling_layer(inputs=conv22, kernel_size=[2, 2], name='pool2')

    # Block 3
    conv31 = layers.conv_bn(inputs=pool2, kernel_size=[
                            3, 3], num_outputs=256, name='conv31', is_training=is_training)
    conv32 = layers.conv_bn(inputs=conv31, kernel_size=[
                            3, 3], num_outputs=256, name='conv32', is_training=is_training)
    pool3 = layers.maxpooling_layer(inputs=conv32, kernel_size=[2, 2], name='pool3')
    drop3 = layers.dropout_layer(inputs=pool3, keep_prob=dropout_keep_prob, name='drop3')

    # Block 4
    conv41 = layers.conv_bn(inputs=drop3, kernel_size=[
                            3, 3], num_outputs=512, name='conv41', is_training=is_training)
    conv42 = layers.conv_bn(inputs=conv41, kernel_size=[
                            3, 3], num_outputs=512, name='conv42', is_training=is_training)
    pool4 = layers.maxpooling_layer(inputs=conv42, kernel_size=[2, 2], name='pool4')
    drop4 = layers.dropout_layer(inputs=pool4, keep_prob=dropout_keep_prob, name='drop4')

    # Block 5
    conv51 = layers.conv_bn(inputs=drop4, kernel_size=[
                            3, 3], num_outputs=1024, name='conv51', is_training=is_training)
    conv52 = layers.conv_bn(inputs=conv51, kernel_size=[
                            3, 3], num_outputs=1024, name='conv52', is_training=is_training)
    drop5 = layers.dropout_layer(inputs=conv52, keep_prob=dropout_keep_prob, name='drop5')

    # ================ U-Net Decoder Section (expansive path) ================ #

    # Block 6
    upsample6 = layers.upsample_layer(inputs=drop5, factor=2, name='upsample6')
    concat6 = layers.concat(inputs1=upsample6, inputs2=conv42, name='concat6')
    conv61 = layers.conv_bn(inputs=concat6, kernel_size=[
                            3, 3], num_outputs=512, name='conv61', is_training=is_training)
    conv62 = layers.conv_bn(inputs=conv61, kernel_size=[
                            3, 3], num_outputs=512, name='conv62', is_training=is_training)
    drop6 = layers.dropout_layer(inputs=conv62, keep_prob=dropout_keep_prob, name='drop6')

    # Block 7
    upsample7 = layers.upsample_layer(inputs=drop6, factor=2, name='upsample7')
    concat7 = layers.concat(inputs1=upsample7, inputs2=conv32, name='concat7')
    conv71 = layers.conv_bn(inputs=concat7, kernel_size=[
                            3, 3], num_outputs=256, name='conv71', is_training=is_training)
    conv72 = layers.conv_bn(inputs=conv72, kernel_size=[
                            3, 3], num_outputs=256, name='conv72', is_training=is_training)
    drop7 = layers.dropout_layer(inputs=conv72, keep_prob=dropout_keep_prob, name='drop7')

    # Block 8
    upsample8 = layers.upsample_layer(inputs=drop7, factor=2, name='upsample8')
    concat8 = layers.concat(inputs1=upsample8, inputs2=conv22, name='concat8')
    conv81 = layers.conv_bn(inputs=concat8, kernel_size=[
                            3, 3], num_outputs=128, name='conv81', is_training=is_training)
    conv82 = layers.conv_bn(inputs=conv81, kernel_size=[
                            3, 3], num_outputs=128, name='conv82', is_training=is_training)

    # Block 9
    upsample9 = layers.upsample_layer(inputs=conv82, factor=2, name='upsample9')
    concat9 = layers.concat(inputs1=upsample9, inputs2=conv12, name='concat9')
    conv91 = layers.conv_bn(inputs=concat9, kernel_size=[
                            3, 3], num_outputs=64, name='conv91', is_training=is_training)
    conv92 = layers.conv_bn(inputs=conv91, kernel_size=[
                            3, 3], num_outputs=64, name='conv92', is_training=is_training)

    # Block 10
    score = layers.conv(inputs=conv92, kernel_size=[
                        1, 1], num_outputs=num_classes, name='score', activation_fn=None)
    logits = tf.reshape(score, (-1, num_classes))

    return logits
