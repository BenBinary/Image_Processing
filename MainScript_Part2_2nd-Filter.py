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
from datetime import datetime
import time
import skimage.measure as measure
import skimage.metrics

#import skimage.measure.compare_ssim
# import skimage


#read the image - just the grayscale
imageToUseName = './InputData/7-Bali-Resorts-RIMBA-1.jpg'
originalImg = cv2.imread(imageToUseName)
img_GRAY = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)

cv2.waitKey()


## Noise
# Add noise with the salt and pepper function
# 1st: define function to create some noise to an image
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image. Replaces random pixels with 0 or 1.
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output





# 2nd: Own function
def possion_noise(image):

    noise_img_width, noise_img_height = image.shape

    for i in range(noise_img_width):
        for j in range(noise_img_height):
            # print(str(image[i, j]))
            bool = random.choice([True, False])
            if bool==True:
                r_add = random.randint(0, 150)
                r_mul = random.randint(0, 10)
                image[i, j] = (image[i, j] + r_add) * r_mul
           
    #vals = len(np.unique(image))
    # vals = 2 ** np.ceil(np.log2(vals))
    # noisy = np.random.poisson(image * vals) / float(vals)
    return image







own_noise_img = possion_noise(img_GRAY)
cv2.namedWindow("Possion Noise Image", cv2.WINDOW_NORMAL)
cv2.imwrite("./OutputData/Noise_Possion_Img.jpg", own_noise_img)
cv2.imshow("Possion Noise Image", own_noise_img)
cv2.waitKey()






## Remove the noise - with 3 different options/ filters

## Writing the results in a text file
f = open("./OutputData/results.txt", 'a')



# 1st filter - Average
f.writelines("## 1. Filter - Average ##\n")
kernel = np.ones((5,5),np.float32)/25 #averaging kernel for 5 x 5 window patch

t = time.time()
blur = cv2.filter2D(own_noise_img, -1, kernel) # changed this 
tmpRunTime = time.time() - t

#illustrate the results
cv2.namedWindow("Averaging Filter", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imwrite("./OutputData/Average_Filter_on_SP.jpg",blur)
cv2.imshow("Averaging Filter", blur )
cv2.resizeWindow("Averaging Filter", 480, 360)


#calculate the similarity between the images
# compute the Structural Similarity Index (SSIM) between the two images

## SSIM
#blur_GRAY = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
(score, _) = skimage.metrics.structural_similarity(img_GRAY, blur, full=True)
print( "The SSIM for the Averaging blur filter time was: {:.8f}".format(tmpRunTime), " seconds. SSIM sccre: {:.4f}".format(score))
f.writelines("The SSIM for the Averaging blur filter time was: {:.8f}".format(tmpRunTime) + " seconds. SSIM sccre: {:.4f}".format(score) + "\n")


## Mean Square Error
sum = 0.0
mse_img_width, mse_img_height = img_GRAY.shape

for i in range(mse_img_width):
    for j in range(mse_img_height):
        difference = img_GRAY[i, j] - blur[i, j]
        sum = sum + difference * difference

mse = sum / (mse_img_height * mse_img_width)
print("The MSE for Averging blur is " + str(mse))
f.writelines("The MSE for Averging blur is " + str(mse) + "\n")

cv2.waitKey()





# 2nd filter - Gaussian
f.writelines("## 2. Filter - Gaussian ## \n")
t = time.time()
blur = cv2.GaussianBlur(own_noise_img, (5,5), 0)
tmpRunTime = time.time() - t

#illustrate the results
cv2.namedWindow("Gauss blur Filter", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imwrite("./OutputData/Gaussian_Filter_on_SP.jpg",blur)
cv2.imshow("Gauss blur Filter", blur )
cv2.resizeWindow("Gauss blur Filter", 480, 360)

# SSIM
#calculate the similarity between the images
# compute the Structural Similarity Index (SSIM) between the two images
#blur_GRAY = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
(score, _) = skimage.metrics.structural_similarity(img_GRAY, blur, full=True)
print( "The SSIM for the Gaussian blur filter time was: {:.8f}".format(tmpRunTime), " seconds. SSIM sccre: {:.4f}".format(score))
f.writelines("The SSIM for the Gaussian blur filter time was: {:.8f}".format(tmpRunTime) + " seconds. SSIM sccre: {:.4f}".format(score) + "\n")

## Mean Square Error
sum = 0.0
mse_img_width, mse_img_height = img_GRAY.shape

for i in range(mse_img_width):
    for j in range(mse_img_height):
        difference = img_GRAY[i, j] - blur[i, j]
        # print("Img gray: " + str(img_GRAY[mse_img_width-1, mse_img_height-1]))
        # print("Difference: " + str(difference))
        sum = sum + difference * difference

mse = sum / (mse_img_height * mse_img_width)
print("The MSE for Gaussian is " + str(mse))
f.writelines("The MSE for Gaussian is " + str(mse)+ "\n")


cv2.waitKey()





# 3rd filter - Bilateral
f.writelines("## 3. Filter - Bilateral ##\n")
t = time.time()
blur = cv2.bilateralFilter(own_noise_img, 9, 5, 5)
tmpRunTime = time.time() - t

#illustrate the results
cv2.namedWindow("Bilateral Filter", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imshow("Bilateral Filter", blur)
cv2.imwrite("./OutputData/bilateral-filter.jpg", blur)
cv2.resizeWindow("Bilateral Filter", 480, 360)


cv2.waitKey()



## SSIM
#calculate the similarity between the images
# compute the Structural Similarity Index (SSIM) between the two images
# blur_GRAY = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
(score, _) = skimage.metrics.structural_similarity(img_GRAY, blur, full=True)
print("The SSIM for the Bilateral blur filter time was: {:.8f}".format(tmpRunTime), " seconds. SSIM sccre: {:.4f}".format(score))
f.writelines("The SSIM for the Bilateral blur filter time was: {:.8f}".format(tmpRunTime) + " seconds. SSIM sccre: {:.4f}".format(score) + "\n")

## Mean Square Error
sum = 0.0
mse_img_width, mse_img_height = img_GRAY.shape

for i in range(mse_img_width):
    for j in range(mse_img_height):
        difference = img_GRAY[i, j] - blur[i, j]
        # print("Img gray: " + str(img_GRAY[mse_img_width-1, mse_img_height-1]))
        # print("Difference: " + str(difference))
        sum = sum + difference * difference

mse = sum / (mse_img_height * mse_img_width)
print("The MSE for Bilateral is " + str(mse))
f.writelines("The MSE for Bilateral is " + str(mse) + "\n")