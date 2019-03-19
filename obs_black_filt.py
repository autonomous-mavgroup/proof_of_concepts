import os
import cv2
import numpy as np


def filt_black(img, b_thresh=220):
    """
    Filters the color black from the image by thresholding the sum of RGB channels

    :param img: RGB image pixel array
    :param b_thresh: black color threshold
    :return: filtered image
    """
    img_sum = np.sum(img, 2)
    black = img_sum < b_thresh

    filt_img = np.ones(img_sum.shape)
    filt_img[black] = 0

    return filt_img


for image in os.listdir('.//pics'):
    img = cv2.imread('.//pics//' + image)
    img = np.rot90(img)

    filt_img = filt_black(img)

    cv2.imshow('image', img)
    cv2.imshow('filt_img', filt_img)
    cv2.waitKey(0)


