import glob
import cv2
from time import sleep
import numpy as np
import matplotlib.pyplot as plt

pics = "pics"
pic = 'pic'
final = pic

images = [cv2.imread(file) for file in glob.glob(final + "/*.jpg")]

def nothing(x):
    pass
#create window
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('real', cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

#create trackbar for window
cv2.createTrackbar('y_m','image',54,255,nothing)
cv2.createTrackbar('y_M','image',86,255,nothing)
cv2.createTrackbar('u_m','image',67,255,nothing)
cv2.createTrackbar('u_M','image',102,255,nothing)
cv2.createTrackbar('v_m','image',74,255,nothing)
cv2.createTrackbar('v_M','image',108,255,nothing)

iterations = 1000

#rotate all images
pos = -1
for image in images:
    pos += 1
    real_img = images[pos]
    # grab the dimensions of the image and calculate the center of the image
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    # rotate the image by 90 degrees
    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    image = cv2.warpAffine(image, M, (w, h))
    real_img = cv2.warpAffine(real_img, M, (w, h))

    for it in range(iterations):
        #create mask and colour filter
        lower = np.array([cv2.getTrackbarPos('y_m','image'),cv2.getTrackbarPos('v_m','image'),cv2.getTrackbarPos('u_m','image')])
        upper = np.array([cv2.getTrackbarPos('y_M','image'),cv2.getTrackbarPos('v_M','image'),cv2.getTrackbarPos('u_M','image')])

        #hard coded for testing purposes
        # lower = np.array([70,70,70])
        # upper = np.array([110,110,110])

        #masking and residual image
        mask = cv2.inRange(image, lower, upper)

        #inverting image
        mask = 255-mask
        res = cv2.bitwise_and(image, image, mask= mask)

        #median filtering
        mask = cv2.medianBlur(mask,5)

        cv2.imshow("image",mask)
        cv2.imshow("real", real_img)

        cv2.waitKey(100)
        #cv2.destroyAllWindows()


