
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
import skimage.metrics


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
    row,col = image.shape
    mean = 0
    var = 0.2
    sigma = var**0.2
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy


img_NOISY_BGR = gs_noise(grayImage)

cv2.imshow('Noisy Images', img_NOISY_BGR)
cv2.waitKey()

	


#a. averaging filter
kernel = np.ones((5,5),np.float32)/25 #averaging kernel for 5 x 5 window patch

blur = cv2.filter2D(img_NOISY_BGR, -1, kernel)

#illustrate the results
cv2.namedWindow("Averaging Filter", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imshow("Averaging Filter", blur )
cv2.resizeWindow("Averaging Filter", 480, 360)

#calculate the similarity between the images
# compute the Structural Similarity Index (SSIM) between the two images
# blur_GRAY = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
(score, _) = skimage.metrics.structural_similarity(grayImage, blur, full=True)
print("The SSIM sccre is: {:.4f}".format(score))

cv2.waitKey()



# Run canny on the correct grayscale and display it
blur_img = blur.astype(np.uint8)
canny_edges = cv2.Canny(blur_img, 10, 10, edges=None, apertureSize=5)
cv2.namedWindow("Canny Edges", cv2.WINDOW_NORMAL)
cv2.imshow('Canny Edges on blur picture', canny_edges)

cv2.waitKey()
