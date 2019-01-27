# import the necessary packages
from keras.models import Model
from keras.layers import Input, Conv2D
from keras.optimizers import RMSprop
from unet.layers import unet_encode, unet_maxpool, unet_decode
from unet.loss import dice_coeff, dice_loss, bce_dice_loss


def unet(input_shape=(128, 128, 3),
         num_classes=1,
         start_filters=64,
         center_filters=1024,
         learning_rate=0.0001):
    '''
    Creating the entire U-Net network
    ---
    Args:
        input_shape:
        num_classes:
        start_filters:
        center_filters:
        learning_rate:
    Returns:
        model
    '''
    num_filters = start_filters
    output_list = [Input(shape=input_shape)]
    encode_temp_list = []

    # while loop stops after 512 filters for unet-128
    while num_filters <= (center_filters / 2):
        x = unet_encode(inputs=output_list[-1], num_filters=num_filters)
        encode_temp_list.append(x)
        y = unet_maxpool(inputs=x)
        output_list.append(y)
        num_filters = num_filters * 2

    # for 1024 filters in center for unet-128
    x = unet_encode(inputs=output_list[-1], num_filters=num_filters)
    output_list.append(x)
    num_filters = int(num_filters / 2)

    # up-sampling starts with 512 filters
    while num_filters >= start_filters:
        x = unet_decode(up_input=output_list[-1], down_input=encode_temp_list.pop(),
                        num_filters=num_filters)
        output_list.append(x)
        num_filters = int(num_filters / 2)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(output_list[-1])

    model = Model(inputs=output_list[0], outputs=classify)

    # set learning rate at 0.0001
    model.compile(optimizer=RMSprop(lr=learning_rate),
                  loss=bce_dice_loss,
                  metrics=[dice_coeff])

    return model
