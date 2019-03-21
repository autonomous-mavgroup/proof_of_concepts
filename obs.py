import glob
import cv2
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from functions import line_fit,ransac_inl,diff

images = [cv2.imread(file) for file in glob.glob("pics/*.jpg")]


def nothing(x):
    pass
#create window
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('real', cv2.WINDOW_NORMAL)

#generate video
[height,width,depth] = np.shape(images[0])




# cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

#create trackbar for window
cv2.createTrackbar('y_m','image',0,255,nothing)
cv2.createTrackbar('y_M','image',181,255,nothing)
cv2.createTrackbar('u_m','image',2,255,nothing)
cv2.createTrackbar('u_M','image',52,255,nothing) #52
cv2.createTrackbar('v_m','image',70,255,nothing)
cv2.createTrackbar('v_M','image',111,255,nothing)

#rotate all images
im_no = 0
sub_samp = False
for image in images:
    image = np.rot90(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    found_points = False

    while(True):
        #create mask and colour filter
        lower = np.array([cv2.getTrackbarPos('y_m','image'),cv2.getTrackbarPos('u_m','image'),cv2.getTrackbarPos('v_m','image')])
        upper = np.array([cv2.getTrackbarPos('y_M','image'),cv2.getTrackbarPos('u_M','image'),cv2.getTrackbarPos('v_M','image')])
        #hard coded for testing purposes
        # lower = np.array([70,70,70])
        # upper = np.array([110,110,110])

        #masking and residual image
        mask = cv2.inRange(image, lower, upper)

        #inverting image
        res = cv2.bitwise_and(image, image, mask= mask)

        #median filtering
        mask = cv2.medianBlur(mask,5)
        #edges = cv2.Canny(mask, 300, 400)
        v_edge_kernel = np.array([[-1,2,-1],
                         [-1,2,-1],
                         [-1,2,-1]])
        
        edges = cv2.filter2D(mask,-1,v_edge_kernel)
        [height,width] = np.shape(edges)
        edge_lines = []
        no_lines = 100
        if sub_samp==True:
                lines = np.array((width-1)*np.random.rand(no_lines,1))
                lines = lines.astype(int)
        elif sub_samp ==False:
                lines = range(width)
        for i in lines:
                for step in range(height-1):
                        if (edges[(height-1-step),i]>0 ):
                                up_green_test = np.count_nonzero( np.array( [mask[(height-1-step)-1,i],mask[(height-1-step)-2,i],mask[(height-1-step)-3,i],mask[(height-1-step)-4,i]] ))
                                # try:
                                #         side_green_test = np.count_nonzero( np.array( [mask[(height-1-step),i-1],mask[(height-1-step)-2,i-2],mask[(height-1-step)-3,i+1],mask[(height-1-step)-4,i+2]] ))
                                if (up_green_test==0):
                                        edge_lines.append([(height-1-step),i])
                                        found_points = True
                                        break
                        if step>int(height*0.4):
                                break

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        crit_points = np.zeros((width))
        x_arr = []
        y_arr = []
        if found_points==True:
                diff_points = diff(edge_lines) 
                [points,depth] = np.shape(diff_points)

                for i in range(points):
                        x_loc = int(diff_points[i,0])
                        y_loc = int(diff_points[i,1])
                        if diff_points[i,2]>0:
                                #image[y_loc,x_loc,:] = [255,0,0]
                                cv2.line(image, (x_loc,y_loc), (x_loc,0), (0,255,0), 1)
                                crit_points[x_loc] = diff_points[i,2]
                                x_arr.append(x_loc)
                                y_arr.append(y_loc)
                        else:
                                image[y_loc,x_loc,:] = [255,0,0]     

        FOV_X = (120/180)*np.pi
        FOV_Y = (80/180)*np.pi
        alt = 1.5 #m
        x_fin = []
        y_fin = []
        for i in range(np.shape(x_arr)[0]):
                theta = (y_arr[i]-height/2)*(FOV_Y/height)   #rad
                psi = (x_arr[i]-width/2)*(FOV_X/width)      #rad
                print(theta)
                d = alt/(np.arctan(theta))
                print(d)
                x_fin.append(d*np.sin(psi))
                y_fin.append(d*np.cos(psi))
                

        plt.clf()
        plt.subplot(121)       
        plt.scatter(x_fin,y_fin)
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        #plt.imshow(mask)
        plt.subplot(122)
        plt.imshow(image)
        #plt.imshow(edges)
    
        #plt.show(block=False)
        plt.savefig('experiment(%i).png' % im_no)
        im_no+=1
        print(im_no)
        plt.pause(0.0001)
        break
        #cv2.destroyAllWindows()

