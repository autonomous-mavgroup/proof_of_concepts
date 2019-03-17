import glob
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from functions import line_fit,ransac_inl
from scipy import signal

images = [cv2.imread(file) for file in glob.glob("pics/*.jpg")]

class Point:
  def __init__(self,x,y):
    self.x = x
    self.y = y


results = np.zeros((len(images),2))
im_number =0
plotting=False
for image in images:
    image = np.rot90(image)
    while(True):
        #create mask and colour filter
        lower = np.array([63,70,74])
        upper = np.array([99,108,111])
        t0 = time.time()
        #lower = np.array([57,70,75])
        #upper = np.array([105,93,104])

        #hard coded for testing purposes
        # lower = np.array([70,70,70])
        # upper = np.array([110,110,110])
        
        #masking and residual image
        mask = cv2.inRange(image, lower, upper)

        #inverting image
        mask = 255-mask
        res = cv2.bitwise_and(image, image, mask= mask)

        #median filtering
        mask = cv2.blur(mask,(5,5))


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
        for line in lines:
            x_sum =0
            no_hits = 0
            for i in range(width):
                if edges[line,i]==255:
                    x_sum += i
                    no_hits +=1
                    #x_loc = i
                    #points.append(Point(x_loc,line))
                    #image[line,x_loc,:] = [0,255,0]
                    break
            if no_hits>0:
                x_loc = int(x_sum/no_hits)
                points.append(Point(x_loc,line))
                image[line,x_loc,:] = [0,255,0]

        if no_hits>0:
            #perform RANSAC line fit
            pix_threshold = 10   #these are counted as inliers
            num_rand_points = 5  #num_points that are considered for every iteration
            iterations = 10       #number of iterations
            best_inliers = 0
            for i in range(iterations):
                rand_points   = np.random.rand(num_rand_points)*(len(points)-1)
                rand_points = rand_points.astype(int) 
                it_points = []  
                for point in rand_points:
                    it_points.append(points[point])
                coefficients_loc = line_fit(it_points)
                inliers = ransac_inl(points,coefficients_loc,pix_threshold)
                if inliers>best_inliers:
                    best_inliers=inliers
                    coefficients = coefficients_loc
        
            t_comp = time.time()-t0
            print("Calculated in :")
            print(t_comp)
            print('image number:')
            print(im_number)
            if(plotting==True):
                num_graph_points = 100
                y_arr = (height*np.array(range(num_graph_points))/num_graph_points)
                y_arr = y_arr.astype(int)
                x_arr = []
                for y_loc in y_arr:
                    x_arr.append(coefficients[2]*y_loc*y_loc+coefficients[1]*y_loc+coefficients[0])
                
                plt.subplot(121)
                plt.imshow(edges)
                plt.subplot(122)
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.plot(x_arr,y_arr)

                # figManager = plt.get_current_fig_manager() 
                #figManager.full_screen_toggle() 


                plt.show(block=False)
                plt.pause(0.00001)
                plt.clf()

            results[im_number,:] = [(coefficients[2]*height*2+coefficients[1]),(coefficients[2]*height*height+coefficients[1]*height+coefficients[0])]
            
            im_number=im_number+1
            break
        else:
            break
#low-pass filter
fs = 30
fc = 10  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(5, w, 'low')
results[:,0] = signal.filtfilt(b, a, results[:,0])
results[:,1] = signal.filtfilt(b, a, results[:,1])

plt.subplot(121)
plt.title('yaw')
plt.plot(results[:,0])
plt.subplot(122)
plt.title('roll')
plt.plot(results[:,1])
plt.show()
        #cv2.waitKey(100000)
        #cv2.destroyAllWindows()

