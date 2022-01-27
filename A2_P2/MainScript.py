
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


# Read the file 
# 1. Task: Read it as BGR and transform it into RGB
imageToUseName = './InputData/crazy_road.jpg'
img_bgr = cv2.imread(imageToUseName)


grayImage = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# Detect Edges with the canny technique
canny_edges = cv2.Canny(grayImage, 10, 10, edges=None, apertureSize=5)
cv2.namedWindow("Canny Edges", cv2.WINDOW_NORMAL)
cv2.imshow('Canny Edges', canny_edges)

cv2.waitKey()


# Gaussian Noise on the grayscale image
def gs_noise(image):
    #gauss noise Gaussian-distributed additive noise.
    row,col,ch= image.shape
    mean = 0
    var = 0.4
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy


img_NOISY_BGR = gs_noise(img_bgr)

cv2.imshow('Noisy Images', img_NOISY_BGR)
cv2.waitKey()

# Remove the noise with a filter of choice and display the correct image


# Run canny on the correct grayscale and display it