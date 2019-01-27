# import the necessary packages
import numpy as np
import cv2
from unet.augment import random_hsv, random_shift_scale_rotate, random_horizontal_flip


def train_generator(input_size, train_id, batch_size):
    '''
    Training generator in keras to get batches of training input images and corresponding
    output masks during training process
    '''
    while True:
        for start in range(0, len(train_id), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(train_id))
            train_batch = train_id[start:end]
            for id in train_batch.values:
                train_dir = 'train/' + id + '.jpg'
                img = cv2.imread(train_dir)
                img = cv2.resize(img, (input_size, input_size))

                train_mask_dir = 'train_masks_png/' + id + '_mask.png'
                mask = cv2.imread(train_mask_dir, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (input_size, input_size))

                img = random_hsv(image=img,
                                 hue_shift_limit=(-50, 50),
                                 sat_shift_limit=(-5, 5),
                                 val_shift_limit=(-15, 15),
                                 u=0.5)

                img, mask = random_shift_scale_rotate(image=img, mask=mask,
                                                      shift_limit=(-0.0625, 0.0625),
                                                      scale_limit=(-0.1, 0.1),
                                                      rotate_limit=(-0, 0),
                                                      borderMode=cv2.BORDER_CONSTANT,
                                                      u=0.5)

                img, mask = random_horizontal_flip(image=img, mask=mask,
                                                   u=0.5)
                mask = np.expand_dims(mask, axis=2)

                x_batch.append(img)
                y_batch.append(mask)

            x_batch = np.array(x_batch, dtype=np.float32) / 255.0
            y_batch = np.array(y_batch, dtype=np.float32) / 255.0

            yield x_batch, y_batch
