import glob
import cv2
from time import sleep
import numpy as np
import matplotlib.pyplot as plt

images = [cv2.imread(file) for file in glob.glob("pic/*.jpg")]


def nothing(x):
    pass
#create window
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('real', cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

#create trackbar for window
cv2.createTrackbar('y_m','image',74,255,nothing)
cv2.createTrackbar('y_M','image',103,255,nothing)
cv2.createTrackbar('u_m','image',62,255,nothing)
cv2.createTrackbar('u_M','image',92,255,nothing)
cv2.createTrackbar('v_m','image',69,255,nothing)
cv2.createTrackbar('v_M','image',86,255,nothing)

#rotate all images

for image in images:
    image = np.rot90(image)


    while(True):
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
        #edges = cv2.Canny(mask, 300, 400)
        v_edge_kernel = np.array([[-1,2,-1],
                         [-1,2,-1],
                         [-1,2,-1]])
        edges = cv2.filter2D(mask,-1,v_edge_kernel)

        cv2.imshow("image",mask)
        cv2.imshow("real", image)
        cv2.imshow("edges",edges)

        cv2.waitKey(100)
        #cv2.destroyAllWindows()

