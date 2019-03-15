import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("pics/1695638921.jpg")
# rotate image
image = np.rot90(image)
# convert to RGB
#imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# convert to HSV
#image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_green = np.array([72, 74, 70])
upper_green = np.array([85, 97, 82])
mask = cv2.inRange(image, lower_green, upper_green)
mask = 255-mask
mask = cv2.medianBlur(mask,5)
result = cv2.bitwise_and(image, image, mask=mask)
# average it out/ remove noise
result = cv2.medianBlur(result, 21)

edges = cv2.Canny(result, 300, 400)

cv2.imshow("original",image)
cv2.imshow("mask", edges)
cv2.imshow("result", result)
cv2.waitKey(1000000)