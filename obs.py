import glob
import cv2
from time import sleep
import numpy as np

images = [cv2.imread(file) for file in glob.glob("pics/*.jpg")]

def nothing(x):
    pass
#create window
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

#create trackbar for window
cv2.createTrackbar('y_m','image',0,255,nothing)
cv2.createTrackbar('y_M','image',255,255,nothing)
cv2.createTrackbar('u_m','image',0,255,nothing)
cv2.createTrackbar('u_M','image',255,255,nothing)
cv2.createTrackbar('v_m','image',0,255,nothing)
cv2.createTrackbar('v_M','image',255,255,nothing)

#rotate all images
for image in images:
    # rotate image
    image = np.rot90(image)

    #create mask and colour filter
    lower = np.array([cv2.getTrackbarPos('y_m', 'image'), cv2.getTrackbarPos('v_m','image'),cv2.getTrackbarPos('u_m','image')])
    upper = np.array([cv2.getTrackbarPos('y_M', 'image'), cv2.getTrackbarPos('v_M','image'),cv2.getTrackbarPos('u_M','image')])
    
    #hard coded for testing purposes
    lower = np.array([70, 70, 70])
    upper = np.array([110, 110, 110])
    
    #masking and residual image
    mask = cv2.inRange(image, lower, upper)

    #inverting image
    mask = 255-mask
    res = cv2.bitwise_and(image, image, mask=mask)

    #median filtering
    mask = cv2.medianBlur(mask, 5)

    cv2.imshow("image", mask)
    cv2.waitKey(1000)
    #cv2.destroyAllWindows()
    

