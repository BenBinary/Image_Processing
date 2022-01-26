###########################################################
###########################################################
#### Part of the Semester Assignment of Benedikt Kurz #####
################# Module: Image Processing ################
##################### Department of Informatics ###########
######## E-Mail: benedikt.kurz002@stud.fh-dortmund.de #####
########## Registration Number:     ###
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

# Read the file 
# 1. Task: Read it as BGR and transform it into RGB
imageToUseName = './InputData/7-Bali-Resorts-RIMBA-1.jpg'
img_bgr = cv2.imread(imageToUseName)

width, height, _ = img_bgr.shape

for i in range(width):
    for j in range(height):
        # Get the rgb values
        bgr_value_blue = img_bgr[i, j, 0]
        bgr_value_green = img_bgr[i, j, 1]
        bgr_value_red = img_bgr[i, j, 2]
        # print("Blue: " + str(bgr_value_blue) 
        #	+ ", Green: " + str(bgr_value_green) 
        #	+ ", Red: " + str(bgr_value_red)) 

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

cv2.imshow("RGB Image", img_rgb)
#cv2.resizeWindow("RGB Image", 480, 360)
cv2.waitKey()

# Plot the graph




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


for i in range(width):
    for j in range(height):
        # Get the rgb values
        rgb_value = img_rgb[i, j, 0]
        #print(rgb_value)


cv2.namedWindow("HSV Image", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imshow("HSV Image", img_hsv)
cv2.imwrite("hsv-image.jpg", img_hsv)
# cv2.resizeWindow("HSV Image", 480, 360)
cv2.waitKey()


## Present now the color channels