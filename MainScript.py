# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2;
from datetime import datetime


#define image name/path
fileNameToUse = 'C:/Users/FHBBook/Desktop/M3_Semester/AT_Image-Procession/image processing/image processing/Lab 1/jinx.jpg'

#read the image
originalImg = cv2.imread(fileNameToUse, 0)



imageToUseName = './InputData/7-Bali-Resorts-RIMBA-1.jpg'
originalImg = cv2.imread(imageToUseName, 0)

p = originalImg.shape
print(p)


#grayImage = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)

# print(gray)


## Function
def solarize(img, thresValue): 

    rows, cols = img.shape

    # Iteratie for loop from 0 to 255
    for i in range(rows):
        for j in range(cols):
            k = originalImg[i,j]
           

            # proves it the pixel gray value is above the threshold
            if (k < thresValue):
                # if so, replaces the value by the negative complement (invert it)
                img[i][j] = -k;
                # print("found sth")

    ## Print it and save the file 
    fname = "Picture_Threshold-" +  str(thresValue) + ".jpg"
    cv2.imwrite(fname, img)
    cv2.imshow("Computed Image", img)


## Apply it with different thresholds like 64, 128 and 192
computedImg = solarize(originalImg, 64)
#computedImg = solarize(originalImg, 128)
#computedImg = solarize(originalImg, 192)

cv2.waitKey()



