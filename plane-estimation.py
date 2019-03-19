import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("pic/91777999.jpg")
# rotate image
image = np.rot90(image)
# convert to RGB
imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# convert to HSV
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_green = np.array([0, 0, 0])
upper_green = np.array([120, 140, 255])
mask = cv2.inRange(image, lower_green, upper_green)
result = cv2.bitwise_and(image, image, mask=mask)
# average it out/ remove noise
result = cv2.medianBlur(result, 21)

edges = cv2.Canny(result, 300, 400)


cv2.imshow("mask", edges)
cv2.imshow("result", result)
cv2.waitKey(0)