# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 09:46:45 2019

@author: RMHanchate2000
"""
# -*- coding: utf-8 -*-


from PIL import Image
import cv2
import numpy as np

#read
im1 = cv2.imread("photo_1.jpg")

# grayscaling and gaussian filtering
im_gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im_gray2 = cv2.GaussianBlur(im_gray1, (5, 5), 0)

# Thresholding
ret, im_th = cv2.threshold(im_gray2, 90, 255, cv2.THRESH_BINARY_INV)

#finding contours
im2, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

#number of images
idx = 0

#bounding rectangles around digits
for ctr in ctrs:
    idx += 1
    x,y,w,h = cv2.boundingRect(ctr)
    roi=im2[y:y+h,x:x+w]
    
    #creating blank image of max dim    
    blank_image = np.zeros((max(w, h), max (w, h), 3), np.uint8)
    
    #for PIL
    cv2.imwrite('D:\\NITK\\Python\\crop\\temp\\' + 'roi.jpg', roi)
    cv2.imwrite('D:\\NITK\\Python\\crop\\temp\\' + 'blank.jpg', blank_image)
    
    #PIL Lazy Operation
    img1 = Image.open('D:\\NITK\\Python\\crop\\temp\\roi.jpg')
    img2 = Image.open('D:\\NITK\\Python\\crop\\temp\\blank.jpg')
    
    #pasting ROIs into blank for retaining aspect ratio
    img2.paste(img1, (int((max(w, h) - w) / 2), int((max(w, h) - h) / 2)))
    
    #coverting object into array
    img3 = np.asarray(img2)
    
    #resizing
    img4 = cv2.resize(img3, (28, 28), interpolation = cv2.INTER_AREA)
    
    #writetime
    cv2.imwrite('D:\\NITK\\Python\\crop\\res\\' + str(idx) + '.jpg', img4)
    
    #showtime
    cv2.imshow("Resulting Image with Rectangular ROIs", img4)
    cv2.waitKey()
    cv2.destroyAllWindows()