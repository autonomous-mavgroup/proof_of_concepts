import cv2
import matplotlib.pyplot as plt
import numpy as np

name = '81578080.jpg'

image = cv2.imread("pic/" + name, 0)
real_img = cv2.imread("pic/" + name)

image = np.rot90(image)



edges = cv2.Canny(image, 20, 30, 1, L2gradient=False)

steps = 2
x_pixel = range(0,np.shape(edges)[1], steps)
y_pixel = np.zeros(len(x_pixel))
difference = np.zeros(len(x_pixel))
for column in x_pixel:
    # y_pixel[column] = np.shape(edges)[0] - np.where(edges[:, column] == 255)[0][-2]
    y_pixel[int(column/steps)] = np.where(edges[:, column] == 255)[0][-1]
    if column != 0:
        difference[int(column/steps)] = abs(y_pixel[int(column/steps)] - y_pixel[int(column/steps)-1])

threshold = 10
closeness = 3
x_obs_dummy = np.where(difference>threshold)[0]
print(x_obs_dummy)
x_obs = []
avoid = True
for i in range(len(x_obs_dummy)-1):
    if x_obs_dummy[i+1] - x_obs_dummy[i] < closeness and avoid:
        x_obs.append(round((x_obs_dummy[i] + x_obs_dummy[i + 1])/2)*steps)
        avoid = False
        print(i, x_obs_dummy[i], avoid)
    elif x_obs_dummy[i+1] - x_obs_dummy[i] < closeness:
        x_obs.append(x_obs_dummy[i]*steps)
        avoid = True
        if i == (len(x_obs_dummy) - 2):
            x_obs.append(x_obs_dummy[i+1]*steps)
    else:
        avoid = True
        if i == (len(x_obs_dummy) - 2):
            x_obs.append(x_obs_dummy[i+1]*steps)

x_obs_center = np.mean(x_obs)
y_obs = max(y_pixel[np.where(difference == max(difference))[0][0]],
        y_pixel[np.where(difference == max(difference))[0][0]-1])

plt.imshow(edges)
plt.plot(x_pixel,y_pixel,'bo', markersize = 1)
plt.hlines(y_obs,0,len(x_pixel)*steps)
plt.vlines(x_obs, 0, y_obs, 'r', linewidth = 1)
plt.plot(x_obs_center,y_obs, "ro", markersize = 3)

plt.show()
