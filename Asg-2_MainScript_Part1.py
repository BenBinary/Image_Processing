###########################################################
###########################################################
#### Part of the Semester Assignment of Benedikt Kurz #####
################# Module: Image Processing ################
##################### Department of Informatics ###########
######## E-Mail: benedikt.kurz002@stud.fh-dortmund.de #####
########## Registration Number: 21390260 ##################
###########################################################
###########################################################

import cv2;
import numpy as np
import random
import time
import skimage.measure as measure
import skimage.metrics
import matplotlib.pyplot as plt
import colorsys
import mpl_toolkits.mplot3d.axes3d as p3

# Read the file 
# 1. Task: Read it as BGR and transform it into RGB
imageToUseName = './InputData/7-Bali-Resorts-RIMBA-1.jpg'
img_bgr = cv2.imread(imageToUseName)

width, height, _ = img_bgr.shape
array_length = width * height
k = 0

r_val = [0] * array_length
g_val = [0] * array_length
b_val = [0] * array_length

for i in range(width):
    for j in range(height):

        # Get the rgb values
        bgr_value_blue = img_bgr[i, j, 0]
        b_val[k] = bgr_value_blue

        bgr_value_green = img_bgr[i, j, 1]
        g_val[k] = bgr_value_green

        bgr_value_red = img_bgr[i, j, 2]
        r_val[k] = bgr_value_red

        k = k + 1
        # print("Blue: " + str(bgr_value_blue) 
        #	+ ", Green: " + str(bgr_value_green) 
        #	+ ", Red: " + str(bgr_value_red)) 

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

cv2.imshow("RGB Image", img_rgb)
#cv2.resizeWindow("RGB Image", 480, 360)
cv2.waitKey()

# Plot the graph
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.scatter(r_val, g_val, b_val, marker='o',  facecolors=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).reshape(-1,3)/255)

ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
rgb_distribution = fig.add_axes(ax)
#cv2.imwrite("rgb_distribution.jpg", rgb_distribution)
#plt.show()
plt.savefig("rgb_distribution.jpg")
# plt save image


# 2. Task: Convert it to HSV

img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

width, height, _ = img_rgb.shape


for i in range(width):
    for j in range(height):
        # Get the rgb values
        bgr_value_blue = img_bgr[i, j, 0]
        bgr_value_green = img_bgr[i, j, 1]
        bgr_value_red = img_bgr[i, j, 2]
        hsv_val = colorsys.rgb_to_hsv(bgr_value_red, bgr_value_green, bgr_value_blue)

        img_hsv[i][i] = hsv_val

        # print("HSV value: " + str(hsv_val))



cv2.namedWindow("HSV Image", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imshow("HSV Image", img_hsv)
cv2.imwrite("hsv-image.jpg", img_hsv)
# cv2.resizeWindow("HSV Image", 480, 360)
cv2.waitKey()


## Present now the color channels