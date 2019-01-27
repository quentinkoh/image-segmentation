# import the necessary packages
import numpy as np
import cv2


def valid_generator(input_size, valid_id, batch_size):
    '''
    Validation generator in keras to get batches of validating input images and corresponding
    output masks during validation process
    '''
    while True:
        for start in range(0, len(valid_id), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(valid_id))
            valid_batch = valid_id[start:end]
            for id in valid_batch.values:
                train_dir = 'train/' + id + '.jpg'
                img = cv2.imread(train_dir)
                img = cv2.resize(img, (input_size, input_size))

                train_mask_dir = 'train_masks_png/' + id + '_mask.png'
                mask = cv2.imread(train_mask_dir, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (input_size, input_size))
                mask = np.expand_dims(mask, axis=2)

                x_batch.append(img)
                y_batch.append(mask)

            x_batch = np.array(x_batch, dtype=np.float32) / 255.0
            y_batch = np.array(y_batch, dtype=np.float32) / 255.0

            yield x_batch, y_batch
