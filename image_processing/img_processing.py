# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 16:48:42 2016

@author: ubuntu
"""

import numpy as np
import cv2

#from matplotlib import pyplot as plt


# Load an color image in grayscale
img = cv2.imread("img/other.jpg")
imgNorth = cv2.imread("img/north.jpg")
imgeqst = cv2.imread("img/east.jpg")
# Transform image to GrayScale or not
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Gradient transformation 
laplacian = cv2.Laplacian(gray_image,cv2.CV_64F)
sobelx = cv2.Sobel(gray_image,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(gray_image,cv2.CV_64F,0,1,ksize=5)


#Canny Edge detection check TIVA tp, seems really relevant !!!
#Hough transormation might give the same insights, TODO check this
edges = cv2.Canny(gray_image,100,200)

#Contours and hierarchy might give insights but not same number contour detected
#Too much preporcessing before use
#ret,thresh = cv2.threshold(gray_image,127,255,0)
#contours,hierarchy = cv2.findContours(thresh,2,1)

#We can also normalize the histogram of images before detecting features
hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]
#Good preprocessing but as in the docs of opencv might loose some orientation
#For relief so we should so use : Disccus about it

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray_image)

#Fourier Transform


dft = cv2.dft(np.float32(gray_image),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
#plt.imshow(magnitude_spectrum, cmap = 'gray')
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
#Numpy plot not working on my computer 
#TODO : check manually plotting features for Fourier transformation


#I think this is only available on OpenCV 3 I got 2.x
#sift = cv2.SIFT()
#kp = sift.detect(gray,None)
#
##result_sift=cv2.drawKeypoints(gray_image,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#sift = cv2.SIFT()
#kp, des = sift.detectAndCompute(gray_image,None)
#




#cv2.imshow('image', img2)#
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()