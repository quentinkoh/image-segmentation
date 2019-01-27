# import the necessary packages
import numpy as np
import cv2


def random_horizontal_flip(image, mask, u=0.5):
    '''
    Shifting image and mask around the y-axis
    ---
    Args:
        image: numpy array by the shape of (height, width, channels)
        mask: numpy array by the shape of (height, width, channels)
        u: threshold value for deciding if image should be flipped around the y-axis or not

    Returns:
        image: flipped imaged
        mask: flipped mask
    '''
    if np.random.random() < u:
        image = cv2.flip(src=image, flipCode=1)
        mask = cv2.flip(src=mask, flipCode=1)

    return image, mask


def random_hsv(image, hue_shift_limit=(-180, 180), sat_shift_limit=(-255, 255),
               val_shift_limit=(-255, 255), u=0.5):
    '''
    Shifting images in its HSV color space and converting it
    back to BGR color space
    ---
    Args:
        image: numpy array by the shape of (height, width, channels)
        hue_shift_limit: range for changing hue
        sat_shift_limit: range for changing saturation
        val_shift_limit: range for changing value
        u: threshold value for deciding if image's HSV value
        should be adjusted or not

    Returns:
        image: shifted image by its HSV but expressed in BGR channels
    '''
    if np.random.random() < u:
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)
        # splitting the image into hue, saturation, value channels
        h, s, v = cv2.split(image)

        # range for changing hue
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)

        # range for changing saturation
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)

        # range for changing value
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)

        # merging individual channels back to hsv image
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(src=image, code=cv2.COLOR_HSV2BGR)

    return image


def random_shift_scale_rotate(image, mask,
                              shift_limit=(-0.0625, 0.0625),
                              scale_limit=(-0.1, 0.1),
                              rotate_limit=(-45, 45),
                              aspect_limit=(0, 0),
                              borderMode=cv2.BORDER_CONSTANT,
                              u=0.5):
    '''
    Shifting, scaling and rotating of images
    ---
    Args:
        image: numpy array by the shape of (height, width, channel)
        mask: numpy array by the shape of (height, width, channels)
        shift_limt: range for shifting image
        scale_limit: range for scaling image
        rotate_limit: range for rotating image
        aspect_limit: range for changing image's aspect ratio
        borderMode: pixel extrapolation method by adding a constant colored border
                            around image
        u: threshold value for deciding if image should be augmented or not

    Returns:
        image, mask
    '''
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])

        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)

        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx,
                                                         height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)

        image = cv2.warpPerspective(image, mat, (width, height),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode,
                                    borderValue=(0, 0, 0,))

        mask = cv2.warpPerspective(mask, mat, (width, height),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=borderMode,
                                   borderValue=(0, 0, 0,))

    return image, mask
