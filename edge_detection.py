import cv2
import matplotlib.pyplot as plt
import numpy

image = cv2.imread("pic/91777999.jpg", 0)
real_img = cv2.imread("pic/91777999.jpg")


(h, w) = image.shape[:2]
center = (w / 2, h / 2)

# rotate the image by 90 degrees
M = cv2.getRotationMatrix2D(center, 90, 1.0)
image = cv2.warpAffine(image, M, (w, h))
real_img = cv2.warpAffine(real_img, M, (w, h))

edges = cv2.Canny(image, 30, 170, 100, L2gradient=False)


# cv2.imshow("real_image", real_img)
cv2.imshow("image", edges)
cv2.waitKey(10000)