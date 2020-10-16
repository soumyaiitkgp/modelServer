# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:35:50 2020

@author: Soumya
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
 

if __name__ == '__main__':
 
    img = cv2.imread("78.jpg")
 
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    points = np.array([[[430,490],[480,490],[480,530],[430,530]]])
 
    #contours
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
 
    res = cv2.bitwise_and(img,img,mask = mask)
    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
 
    ## crate the white background of the same size of original image
    wbg = np.ones_like(img, np.uint8)*255
    cv2.bitwise_not(wbg,wbg, mask=mask)
    # overlap the resulted cropped image on the white background
    dst = wbg+res
 
    
    plt.imshow(img, cmap='gray')
    plt.show()
    plt.imshow(res, cmap='gray')
    plt.show()
    plt.imshow(dst, cmap='gray')
    plt.show()
    plt.imshow(cropped, cmap='gray')
    plt.show()
    
    
    from scipy import ndimage

#rotation angle in degree
    rotated = ndimage.rotate(cropped, 4, reshape=(False))
    plt.imshow(rotated, cmap = 'gray')
    plt.show()

    
    #adjust the cropped image by cropping again
    rotated = rotated[7:29, 15:45]

    #Added Bilateral filter
    rotated = cv2.bilateralFilter(rotated,255, 120, 10000)
    
    plt.imshow(rotated, cmap = 'gray')
    plt.show()  

    #convert to gray image
    image = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    plt.imshow(gray, cmap = 'gray')
    plt.show()

    #Binary conversion of the image
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    plt.imshow(binary, cmap = 'gray')
    plt.show()

    # set kernel as 3x3 matrix from numpy
    kernel = np.ones((3,3), np.uint8)       
    #Create erosion and dilation image from the original image
    dilation_image = cv2.dilate(binary, kernel, iterations=1)
    dilation_image = binary
    plt.imshow(dilation_image, cmap = 'gray')
    plt.show()
    
    im1 = dilation_image[0:22, 0:9]
    plt.imshow(im1, cmap = 'gray')
    plt.show()
    im2 = dilation_image[0:22, 9:18]
    plt.imshow(im2, cmap = 'gray')
    plt.show()    
    im3 = dilation_image[0:22, 21:30]
    plt.imshow(im3, cmap = 'gray')
    plt.show() 
    im1 = cv2.dilate(im1, kernel, iterations=1)
    plt.imshow(im1, cmap = 'gray')
    plt.show()
    im2 = cv2.dilate(im2, kernel, iterations=1)
    plt.imshow(im2, cmap = 'gray')
    plt.show()
    im3 = cv2.dilate(im3, kernel, iterations=1)
    plt.imshow(im3, cmap = 'gray')
    plt.show()
    
    #--------------------------------OCR-----------------------------------------
    
    digits = []
    DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 0, 1): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
    }
    
    sum1 = 0
    sum4 = 0
    sum7 = 0
    i = 0    
    while i < 4:
        j = 0
        while j < 4:
            sum1 += im1[i,j+3]
            sum4 += im1[i+8,j+3]
            sum7 += im1[i+18,j+3]
            j = j + 1
        i = i + 1
    t = i*j
    avg = sum1/(255*t)
    if avg > 0.5:
        st1 = 1
    else:
        st1 = 0
    avg = sum4/(255*t)
    if avg > 0.5:
        st4 = 1
    else:
        st4 = 0
    avg = sum7/(255*t)
    if avg > 0.4:
        st7 = 1
    else:
        st7 = 0
    
    sum2 = 0
    sum3 = 0
    sum5 = 0
    sum6 = 0
    i = 0    
    while i < 5:
        j = 0
        while j < 3:
            sum2 += im1[i+4,j]
            sum3 += im1[i+4,j+6]
            sum5 += im1[i+13,j]
            sum6 += im1[i+13,j+6]
            j = j + 1
        i = i + 1
    t = i*j
    avg = sum2/(255*t)
    if avg > 0.5:
        st2 = 1
    else:
        st2 = 0
    avg = sum3/(255*t)
    if avg > 0.5:
        st3 = 1
    else:
        st3 = 0
    avg = sum5/(255*t)
    if avg > 0.5:
        st5 = 1
    else:
        st5 = 0
    avg = sum6/(255*t)
    if avg > 0.5:
        st6 = 1
    else:
        st6 = 0
    #print(st1,st2,st3,st4,st5,st6,st7)
    on = [st1,st2,st3,st4,st5,st6,st7]
    digit1 = DIGITS_LOOKUP[tuple(on)]
    digits.append(digit1)
    #print(digits,'digit')
    
    #===================---------Digit two------------===============
    sum1 = 0
    sum4 = 0
    sum7 = 0
    i = 0    
    while i < 4:
        j = 0
        while j < 4:
            sum1 += im2[i,j+3]
            sum4 += im2[i+8,j+3]
            sum7 += im2[i+17,j+3]
            j = j + 1
        i = i + 1
    t = i*j
    avg = sum1/(255*t)
    if avg > 0.5:
        st1 = 1
    else:
        st1 = 0
    avg = sum4/(255*t)
    if avg > 0.5:
        st4 = 1
    else:
        st4 = 0
    avg = sum7/(255*t)
    if avg > 0.5:
        st7 = 1
    else:
        st7 = 0
    
    sum2 = 0
    sum3 = 0
    sum5 = 0
    sum6 = 0
    i = 0    
    while i < 5:
        j = 0
        while j < 3:
            sum2 += im2[i+4,j]
            sum3 += im2[i+4,j+6]
            sum5 += im2[i+13,j]
            sum6 += im2[i+13,j+6]
            j = j + 1
        i = i + 1
    t = i*j
    avg = sum2/(255*t)
    if avg > 0.5:
        st2 = 1
    else:
        st2 = 0
    avg = sum3/(255*t)
    if avg > 0.5:
        st3 = 1
    else:
        st3 = 0
    avg = sum5/(255*t)
    if avg > 0.5:
        st5 = 1
    else:
        st5 = 0
    avg = sum6/(255*t)
    if avg > 0.5:
        st6 = 1
    else:
        st6 = 0
    #print(st1,st2,st3,st4,st5,st6,st7)
    on = [st1,st2,st3,st4,st5,st6,st7]
    digit2 = DIGITS_LOOKUP[tuple(on)]
    digits.append(digit2)
    #print(digits,'digit')
    
    
    #=======-------------Digit3------========================
    
    sum1 = 0
    sum4 = 0
    sum7 = 0
    i = 0    
    while i < 4:
        j = 0
        while j < 4:
            sum1 += im3[i,j+3]
            sum4 += im3[i+8,j+3]
            sum7 += im3[i+17,j+3]
            j = j + 1
        i = i + 1
    t = i*j
    avg = sum1/(255*t)
    if avg > 0.5:
        st1 = 1
    else:
        st1 = 0
    avg = sum4/(255*t)
    if avg > 0.5:
        st4 = 1
    else:
        st4 = 0
    avg = sum7/(255*t)
    if avg > 0.5:
        st7 = 1
    else:
        st7 = 0
    
    sum2 = 0
    sum3 = 0
    sum5 = 0
    sum6 = 0
    i = 0    
    while i < 5:
        j = 0
        while j < 3:
            sum2 += im3[i+4,j]
            sum3 += im3[i+4,j+6]
            sum5 += im3[i+13,j]
            sum6 += im3[i+13,j+6]
            j = j + 1
        i = i + 1
    t = i*j
    avg = sum2/(255*t)
    if avg > 0.5:
        st2 = 1
    else:
        st2 = 0
    avg = sum3/(255*t)
    if avg > 0.5:
        st3 = 1
    else:
        st3 = 0
    avg = sum5/(255*t)
    if avg > 0.5:
        st5 = 1
    else:
        st5 = 0
    avg = sum6/(255*t)
    if avg > 0.5:
        st6 = 1
    else:
        st6 = 0
    #print(st1,st2,st3,st4,st5,st6,st7)
    on = [st1,st2,st3,st4,st5,st6,st7]
    digit3 = DIGITS_LOOKUP[tuple(on)]
    digits.append(digit3)
    #print(digits,'digit')
    print(u"{}{}.{} \u00b0C".format(*digits))
    
#----------------------------------------Writing detected values-------------------------------------


import csv
with open('TempOCR.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "amount"])
    writer.writerow([1,u"{}{}.{} \u00b0C".format(*digits)])
    
#====================================================END=======================