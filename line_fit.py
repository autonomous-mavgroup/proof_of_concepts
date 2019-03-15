import glob
import cv2
from time import sleep
import numpy as np
import matplotlib.pyplot as plt

images = [cv2.imread(file) for file in glob.glob("pic/*.jpg")]

def nothing(x):
    pass

class Point:
  def __init__(self,x,y):
    self.x = x
    self.y = y


#create window
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

#create trackbar for window
cv2.createTrackbar('y_m','image',74,255,nothing)
cv2.createTrackbar('y_M','image',103,255,nothing)
cv2.createTrackbar('u_m','image',62,255,nothing)
cv2.createTrackbar('u_M','image',92,255,nothing)
cv2.createTrackbar('v_m','image',69,255,nothing)
cv2.createTrackbar('v_M','image',94,255,nothing)

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

        #compute edges
        edges = cv2.Canny(mask, 300, 400)

        #obtain lines for line fit
        #image format in this file: image[y,x,[RGB]], starting from left-top corner
        [height,width,depth] = np.shape(image)

        lower_fraction = 0.3 #means that only 30% of the image (seen from the bottom) will be used for the line fit

        no_lines = 100 #number of lines to use for fitting
        if no_lines>int(lower_fraction*height):
            no_lines = int(lower_fraction*height)
        lines = np.array(lower_fraction*(height)*np.random.rand(no_lines,1))
        lines = (height-1)-lines.astype(int)
        points = []
        coefficients = [0,0,0]
        for line in lines:
            x_sum =0
            no_hits = 0
            for i in range(width):
                if edges[line,i]==255:
                    x_sum += i
                    no_hits +=1
            if no_hits>0:
                x_loc = int(x_sum/no_hits)
                points.append(Point(x_loc,line))
                image[line,x_loc,:] = [0,255,0]
        #perform line fit
        no_points = len(points)
        sum_y =0.
        sum_x = 0.
        sum_xy = 0.
        xum_xy = 0.
        sum_x2 = 0.
        sum_x2y =0.
        sum_x3 = 0.
        sum_x4 = 0.
        S_xx = 0.
        S_xy = 0.
        Sxx2 = 0. 
        S_x2y = 0.
        S_x2x2 = 0. 
        avg_y = 0. 
        avg_x = 0. 

        for i in range(no_points):
            x2 = points[i].y * points[i].y
            sum_y   += points[i].x
            sum_x   += points[i].y
            sum_xy  += points[i].y * points[i].x
            sum_x2  += x2
            sum_x2y += x2 * points[i].x
            sum_x3  += points[i].y * x2
            sum_x4  += x2 * x2
        
        avg_y = sum_y / no_points
        avg_x = sum_x / no_points

        S_xx   = sum_x2  - sum_x  * avg_x
        S_xy   = sum_xy  - sum_x  * avg_y
        S_xx2  = sum_x3  - sum_x2 * avg_x
        S_x2y  = sum_x2y - sum_x2 * avg_y
        S_x2x2 = sum_x4  - sum_x2 * sum_x2 / no_points
        scaler = 1. / (S_xx* S_x2x2 - S_xx2*S_xx2)

        coefficients[2] = (S_x2y * S_xx - S_xy * S_xx2) * scaler
        coefficients[1] = (S_xy * S_x2x2 - S_x2y * S_xx2) * scaler
        coefficients[0] = (sum_y - coefficients[1] * sum_x - coefficients[2] * sum_x2) / no_points

        num_graph_points = 100
        y_arr = (height*np.array(range(num_graph_points))/num_graph_points)
        y_arr = y_arr.astype(int)
        x_arr = []
        for y_loc in y_arr:
            x_arr.append(coefficients[2]*y_loc*y_loc+coefficients[1]*y_loc+coefficients[0])
        
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.plot(x_arr,y_arr)
        plt.show()
        sleep(5)
        #cv2.waitKey(100000)
        #cv2.destroyAllWindows()

