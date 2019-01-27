# import the necessary packages
import keras.backend as K
from keras.losses import binary_crossentropy


def dice_coeff(y_true, y_pred):
    '''
    Dice coefficient to measure comparison of the pixel-wise agreement between a predicted
    segmentation and its corresponding ground truth.

    Note that a Laplace smoothing factor (or also known as additive smoothing) has been
    added to reduce over-fitting as the entire Dice coefficient value is made larger and thus,
    lower loss, implying that convergence can be achieved faster and avoiding too many
    training iterations.
    ---
    Args:
        y_true: ground truth mask
        y_pred: predicted mask

    Returns:
        score
    '''
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return score


def dice_loss(y_true, y_pred):
    '''
    Dice loss to measure the loss due to pixel-wise disagreement between a predicted segmentation
    and its corresponding ground truth.
    ---
    Args:
        y_true: ground truth mask
        y_pred: predicted mask

    Returns:
        loss
    '''
    loss = 1 - dice_coeff(y_true, y_pred)

    return loss


def bce_dice_loss(y_true, y_pred):
    '''
    Function combines both binary cross-entropy loss and dice loss for computing total loss
    between a predicted segmentation and its corresponding ground truth.
    ---
    Args:
        y_true: ground truth mask
        y_pred: predicted mask

    Returns:
        total loss
    '''
    total_loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    return total_loss
