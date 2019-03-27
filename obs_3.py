import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import radians, degrees
import time

images = [cv2.imread(file) for file in glob.glob("pics/*.jpg")]

y_threshold = 0.5        # percentage of y-coordinates not considered for analysis
pitch = 0                # deg (pitch down is positive)
obs_threshold = 1        # m
x_step = 2               # step-size for checking pixel columns
alt = 1.5                # vehicle altitude in m
lateral_per = 0.2        # percentage of the image on the left or right side considered to be the "lateral"
lateral_threshold = 5    # number of intrusion points in the lateral position to consider obstacle presence
total_threshold = 30      # number of intrusion points in the complete image to consider obstacle presence
FOV_X = (120/180)*np.pi
FOV_Y = (80/180)*np.pi  
resize_factor = 0.2      # Reduce quality of the image

min_dist = alt/np.tan(FOV_Y/2+pitch)    # distance from the drone in the z-direction when point is at the image bottom
for image in images:
    image = np.rot90(image)
    [height_or, width, depth_or] = np.shape(image)
    height = int(height_or - height_or * y_threshold)   # new height
    image = image[height_or-height:, :, :]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Making and residual image
    image = cv2.resize(image, (int(width*resize_factor), int(height*resize_factor)))
    mask = cv2.inRange(image, np.array([0, 2, 70]), np.array([181, 52, 111]))
    edges = cv2.medianBlur(mask, 11)

    # Median filtering
    [height, width] = np.shape(edges)
    edge_lines = np.zeros((len(range(0, width, x_step)), 2))

    # Appending points that are part of a line
    j = -1
    for i in range(0, width, x_step):
        j += 1
        for step in range(height-1):
            y_loc = height-1-step
            if edges[y_loc, i] == 0:
                up_green_test = np.sum(mask[y_loc-4:y_loc-1, i])
                if not up_green_test:
                        edge_lines[j, :] = [y_loc+1, i]
                        cv2.line(image, (i, y_loc+1), (i, 0), (0, 255, 0), 1)
                        break
            if step > int(height*0.9):
                edge_lines[j, :] = [1, i]
                cv2.line(image, (i, 1), (i, 0), (0, 255, 0), 1)
                break


    x_arr = edge_lines[:, 1]
    y_arr = edge_lines[:, 0]

    y_ratio = FOV_Y/height_or

    y_obs_left = []
    y_obs_right = []
    y_fin = np.zeros(np.shape(x_arr)[0])
    for i in range(np.shape(x_arr)[0]):
        theta = radians(pitch) + (y_threshold*height_or + y_arr[i]/resize_factor-height_or/2)*y_ratio   # rad (radial position of a pixel in the height direction)
        d = alt/np.tan(theta)
        y_fin[i] = d
        if lateral_per*width > x_arr[i]:
            y_obs_left.append(d)
        elif width - lateral_per*width < x_arr[i]:
            y_obs_right.append(d)

    n_obs = len(list(filter(lambda x: x < obs_threshold or x == min_dist, y_fin)))
    n_obs_left = len(list(filter(lambda x: x < obs_threshold or x == min_dist, y_obs_left)))
    n_obs_right = len(list(filter(lambda x: x < obs_threshold or x == min_dist, y_obs_right)))
    print(n_obs, n_obs_left, n_obs_right)

    if n_obs_left > lateral_threshold and n_obs_left > n_obs_right:
        print("OBSTACLE LEFT, TURN RIGHT")
    elif n_obs_right > lateral_threshold and n_obs_left < n_obs_right:
        print("OBSTACLE RIGHT, TURN LEFT")
    elif n_obs > total_threshold and n_obs_left > n_obs_right:
        print("OBSTACLE OR EDGE AHEAD, TURN RIGHT")
    elif n_obs > total_threshold and n_obs_left < n_obs_right:
        print("OBSTACLE OR EDGE AHEAD, TURN LEFT")
    else:
        print("CONTINUE IN THE SAME DIRECTION")

    plt.imshow(image)
    plt.show()
