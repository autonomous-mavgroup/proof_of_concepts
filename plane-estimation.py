import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("pics/77978111.jpg")
# rotate image
image = np.rot90(image)
# convert to RGB
imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(imageRGB)
plt.show()
# convert to HSV
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
plt.imshow(image)
plt.show()
lower_green = np.array([25, 0, 0])
upper_green = np.array([110, 255, 255])
mask = cv2.inRange(image, lower_green, upper_green)
result = cv2.bitwise_and(imageRGB, imageRGB, mask=mask)
# average it out/ remove noise
result = cv2.medianBlur(result, 3)
plt.imshow(result)
# TODO: closing seems not to work
plt.show(block=False)
plt.close('close')