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


def find_safe_vertical(sum_arr, line_skip=1):
    """
    Finds longest horizontal sequence of 0s in 2D array and returns the index
    of the middle of said sequence

    :param sum_arr: 2D array containing zeros
    :param line_skip: tunes how many lines the func will search
    :return: row, col index of middle of longest sequence
    """
    y_tot, _ = sum_arr.shape
    seq_len = 0
    seq_start = 0
    seq_end = 0
    seq_line = 0

    for line in range(0, y_tot, line_skip):
        counter = 0
        start = 0
        for idx in range(len(sum_arr[line, :])):
            if counter == 0 and (sum_arr[line, idx] == 0):
                start = idx
                counter += 1
            elif sum_arr[line, idx] == 0:
                counter += 1
            else:
                if counter > seq_len:
                    seq_start = start
                    seq_end = idx - 1
                    seq_len = counter
                    seq_line = line
                    counter = 0
                else:
                    counter = 0
        else:
            if counter > seq_len:
                seq_start = start
                seq_end = idx - 1
                seq_len = counter
                seq_line = line

    return seq_line, (seq_start + seq_end) // 2


for image in os.listdir('.//pics'):
    img = cv2.imread('.//pics//' + image)
    img = np.rot90(img)

    filt_img = filt_black(img, b_thresh=225)
    white_sum = find_white(filt_img, grid_size=(30, 54))

    row, col = find_safe_vertical(white_sum)

    cv2.imshow('image', img)
    cv2.imshow('filt_img', filt_img)
    cv2.waitKey(0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(white_sum, cmap='gnuplot2')
    ax.add_patch(plt.Circle((col, row), 0.5, color='r', fill=False))
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()

