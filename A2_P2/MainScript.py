
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


## Generic
## Writing the results in a text file
f = open("./OutputData/results.txt", 'a')


### 1 ###
# Read the file 
# 1. Task: Read it as BGR and transform it into RGB
imageToUseName = './InputData/crazy_road.jpg'
img_bgr = cv2.imread(imageToUseName)


grayImage = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

### 2 ###
# Detect Edges with the canny technique
canny_edges_original = cv2.Canny(grayImage, 10, 10, edges=None, apertureSize=5)
cv2.namedWindow("Canny Edges", cv2.WINDOW_NORMAL)
cv2.imshow('Canny Edges', canny_edges_original)
cv2.imwrite("./OutputData/canny_edges_original.jpg", canny_edges_original)
cv2.waitKey()

### 3 ### 
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


## Apply the gaussian noise on the file
img_NOISY = gs_noise(grayImage)
cv2.imshow('Noisy Images', img_NOISY)
cv2.imwrite("./OutputData/image_noisy.jpg", img_NOISY)
cv2.waitKey()




### 4 ###
## Apply averaging filter on the noise picutre
kernel = np.ones((5,5),np.float32)/25 #averaging kernel for 5 x 5 window patch
average_filtered = cv2.filter2D(img_NOISY, -1, kernel)
#illustrate the results
cv2.namedWindow("Averaging Filter", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imshow("Averaging Filter", average_filtered)
cv2.imwrite("./OutputData/average_filtered.jpg", average_filtered)
cv2.waitKey()




### 5 ###
# Run canny on the correct grayscale and display it
blur_img = average_filtered.astype(np.uint8)
canny_edges = cv2.Canny(blur_img, 10, 10, edges=None, apertureSize=5)
cv2.namedWindow("Canny Edges", cv2.WINDOW_NORMAL)
cv2.imshow('Canny Edges on average_filteredpicture', canny_edges)
cv2.imwrite("./OutputData/canny_edges_noisy.jpg", canny_edges)
cv2.waitKey()





### 6 ###
# calculate the similarity between the images
f.writelines("## 6. Compare Canny and normal gray pictures ##\n")


# 1st approach: Calculate the similiarities with SSIM
(score, _) = skimage.metrics.structural_similarity(grayImage, average_filtered, full=True)
print("6-1: The SSIM sccre is: {:.4f}".format(score))
f.writelines("6-1: The SSIM sccre is: {:.4f}".format(score) + "\n")

# 2nd approach: 
width, height = grayImage.shape
counter = 0
pixels = 0

for i in range(width):
    for j in range(height):
        pixels = pixels + 1
        counter = grayImage[width-1, height-1] - average_filtered[width-1, height-1] 

result = counter / pixels

print("6-2: The result of the second approach is: " + str(result) + "\n")
f.writelines("6-2: The result of the second approach is: " + str(result) + "\n")





### 7 ###
# Morphology to detect the edges
f.writelines("## 7. Morphology to detect the edges ##" + "\n")


#apply a threshold to the image
th, img = cv2.threshold(img_NOISY, 120,255, cv2.THRESH_BINARY)
kernel = np.ones((3,3),np.uint8)

cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
cv2.imshow('Threshold', img)
cv2.waitKey()

#1st method: Erosion
erosion = cv2.erode(img_NOISY, kernel, iterations = 1)
cv2.namedWindow("Erosion", cv2.WINDOW_NORMAL)
cv2.imshow('Erosion', erosion)
cv2.imwrite("./OutputData/erosion.jpg", erosion)
cv2.waitKey()

counter = 0
pixels = 0

for i in range(width):
    for j in range(height):
        pixels = pixels + 1
        counter = erosion[width-1, height-1] - canny_edges_original[width-1, height-1] 

result = counter / pixels
print("7-1: The result of the erosion approach is: " + str(result))
f.writelines("7-1: The result of the erosion approach is: " + str(result) + "\n")


#2nd method: Dilation
dilation = cv2.dilate(img, kernel, iterations = 1)
cv2.namedWindow("Dilation", cv2.WINDOW_NORMAL)
cv2.imshow('Dilation', dilation)
cv2.imwrite("./OutputData/dilation.jpg", dilation)
cv2.waitKey()

counter = 0
pixels = 0

for i in range(width):
    for j in range(height):
        pixels = pixels + 1
        counter = dilation[width-1, height-1] - canny_edges_original[width-1, height-1] 

result = counter / pixels
print("7-2: The result of the delation approach is: " + str(result))
f.writelines("7-2: The result of the delation approach is: " + str(result) + "\n")