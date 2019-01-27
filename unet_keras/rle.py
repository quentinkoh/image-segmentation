# import the necessary packages
import numpy as np
import keras.backend as K


def run_length_encode(mask):
    '''
    Run length encode our output masks for submission
    ---
    Args:
        mask: numpy array with label 1 - mask (i.e. car) and 0 - background

    Returns:
        run length encode
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])

    return rle
