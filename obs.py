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
    # grab the dimensions of the image and calculate the center
    # of the image
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    
    # rotate the image by 90 degrees
    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    image = cv2.warpAffine(image, M, (w, h))

    #create mask and colour filter
    lower = np.array([cv2.getTrackbarPos('y_m','image'),cv2.getTrackbarPos('v_m','image'),cv2.getTrackbarPos('u_m','image')])
    upper = np.array([cv2.getTrackbarPos('y_M','image'),cv2.getTrackbarPos('v_M','image'),cv2.getTrackbarPos('u_M','image')])
    
    mask = cv2.inRange(image, lower, upper)
    res = cv2.bitwise_and(image,image, mask= mask)

    cv2.imshow("image",res)
    cv2.waitKey(1)
    #cv2.destroyAllWindows()
    sleep(1)
    

