#edit 1: specified imageDir to read all the images we have for testing,
#
#so now, it will read from imageDir, and check only the images that end in the extensions listed.
#also, we need to decide whether we're using os.path
#@author: NegaMage
#PS: left notes for code so you can read through to understand. 
#May have cluttered it a bit.


from PIL import Image
import cv2
import numpy as np
import os, os.path

#replace with the dir you use. Remember to double backslash
imageDir="C:\\Users\\feyaz\\.spyder-py3\\MNIST"

valid_extensions = [".jpg", ".jpeg", ".png"]


#this code will go through the directory and add every image file to 
#a list image_path_list, initialised below

image_path_list = []

for file in os.listdir(imageDir):
    #split-ext will read the file name, and split the name and extension
    #and return both into a tuple called a pair.
    ext = os.path.splitext(file)[1]
    if ext.lower() not in valid_extensions:
        continue
    image_path_list.append(os.path.join(imageDir, file))

"""
TRAINING CODE HERE
@author: Ankush, Adithya, Vignesh


"""



"""
TESTING CODE

read, resize and box images
@author: RMHanchante2000


"""
#read from home dir.
#
im1 = cv2.imread("photo_1.jpg")

# grayscaling and gaussian filtering
im_gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im_gray2 = cv2.GaussianBlur(im_gray1, (5, 5), 0)

# Thresholding
ret, im_th = cv2.threshold(im_gray2, 90, 255, cv2.THRESH_BINARY_INV)

#finding contours
im2, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

#number of numbers in photo_1.jpg
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
    
    #writing
    cv2.imwrite('D:\\NITK\\Python\\crop\\res\\' + str(idx) + '.jpg', img4)
    
    #showing image
    cv2.imshow("Resulting Image with Rectangular ROIs", img4)
    cv2.waitKey()
    cv2.destroyAllWindows()


"""
Now, all the numbers from photo_1.jpg are in boxes as 1.jpg, 2.jpg,...
Implementing KNN here for ROIs.
"""

