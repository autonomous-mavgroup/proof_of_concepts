import glob
import cv2
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from functions import line_fit,ransac_inl,diff

images = [cv2.imread(file) for file in glob.glob("single_pic/*.jpg")]


def nothing(x):
    pass
#create window
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

#generate video
[height,width,depth] = np.shape(images[0])




# cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

#create trackbar for window
# cv2.createTrackbar('y_m','image',0,255,nothing)
# cv2.createTrackbar('y_M','image',181,255,nothing)
# cv2.createTrackbar('u_m','image',2,255,nothing)
# cv2.createTrackbar('u_M','image',52,255,nothing) #52
# cv2.createTrackbar('v_m','image',70,255,nothing)
# cv2.createTrackbar('v_M','image',111,255,nothing)
cv2.createTrackbar('y_m','image',69,255,nothing)
cv2.createTrackbar('y_M','image',174,255,nothing)
cv2.createTrackbar('u_m','image',98,255,nothing)
cv2.createTrackbar('u_M','image',156,255,nothing) #52
cv2.createTrackbar('v_m','image',84,255,nothing)
cv2.createTrackbar('v_M','image',176,255,nothing)

#rotate all images
im_no = 0
sub_samp = False
for image in images:
    image = np.rot90(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV ) 
    found_points = False

    while(True):
        #create mask and colour filter
        print(np.shape(image))
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)

        lower = np.array([cv2.getTrackbarPos('y_m','image'),cv2.getTrackbarPos('u_m','image'),cv2.getTrackbarPos('v_m','image')])
        upper = np.array([cv2.getTrackbarPos('y_M','image'),cv2.getTrackbarPos('u_M','image'),cv2.getTrackbarPos('v_M','image')])
        #hard coded for testing purposes
        # lower = np.array([70,70,70])
        # upper = np.array([110,110,110])

        #masking and residual image
        mask = cv2.inRange(image, lower, upper)

        #plt.clf()
        cv2.imshow('image',mask)
        #plt.imshow(edges)
    
        
        #plt.pause(0.0001)
        cv2.waitKey(5)
        #cv2.destroyAllWindows()

