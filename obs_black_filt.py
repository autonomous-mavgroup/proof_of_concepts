import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def find_white(filt_arr, grid_size=(7, 10)):
    """
    Sub-samples a 2D array into a rougher 2D array containing the sum of the values
    corresponding to the patch of the original array considered

    :param filt_arr: 2D array of 1s and 0s
    :param grid_size: shape of outputted 2D array
    :return: 2D array
    """
    y, x = filt_arr.shape
    y_step = y // grid_size[0]
    x_step = x // grid_size[1]
    grid = np.empty(grid_size)

    for j in range(grid_size[1]):
        for i in range(grid_size[0]):
            grid[i, j] = np.sum(filt_arr[i*y_step:(i+1)*y_step, j*x_step:(j+1)*x_step])

    return grid


for image in os.listdir('.//pics'):
    img = cv2.imread('.//pics//' + image)
    img = np.rot90(img)

    filt_img = filt_black(img, b_thresh=225)
    white_sum = find_white(filt_img, grid_size=(13, 18))

    cv2.imshow('image', img)
    cv2.imshow('filt_img', filt_img)
    cv2.waitKey(0)

    plt.figure()
    plt.imshow(white_sum, cmap='gnuplot2')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()

