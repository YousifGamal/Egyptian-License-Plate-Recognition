from commonfunctions import *
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv

from skimage.color import rgb2gray
from skimage.morphology import erosion,dilation,skeletonize, thin
from skimage import measure
from skimage.draw import rectangle
from skimage import measure
from PIL import Image
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt
from skimage.morphology import reconstruction
from scipy.signal import convolve2d
from scipy.signal import medfilt
from scipy import ndimage, misc
from scipy.ndimage import interpolation as inter
from timeit import default_timer as timer
import cv2


import numpy as np
import colorsys
import argparse

start = timer()

Originalrgb  =io.imread('New set/done/3.jpg')
Original  = rgb2gray(Originalrgb)
Original2  = rgb2gray(Originalrgb)

img = cv2.imread('New set/done/3.jpg',0)

show_images([img])



Original=sobel(Original)

Original=np.copy(Original)

#Dilation
Dilation= dilation(Original)

#show_images([Original,Dilation ],["Original","Dilation"])

#Filling
seed = np.copy(Dilation)
seed[1:-1,1:-1] = Dilation.max()
mask = Dilation

filled = reconstruction(seed, mask, method='erosion')
filled = filled >0.14
#show_images([filled])




#Erosion
filled = np.copy(filled)
rosion = erosion(filled)
rosion = erosion(rosion)


#show_images([filled,rosion])


seed = np.copy(rosion)
seed[1:-1,1:-1] = rosion.max()
mask = rosion
filled = reconstruction(seed, mask, method='erosion')
filled = filled >0.14
#smoothing with median filter
result = ndimage.median_filter(filled, size=9)
result = ndimage.median_filter(result, size=9)
result = ndimage.median_filter(result, size=9)



#show_images([result])


#Find contours
result2 = result.astype(np.uint8)
result2*=255

copy = Originalrgb.copy()
Img,contours, hier  = cv2.findContours(result2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


#Removing the contours with small area
roi = []
arrcon=[]
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)

    if (cv2.contourArea(c) > 600  ):
        # get the min area rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        arrcon.append(c)


#Remove contours with corners more than 5
arrconchoosen=[]
for cnt in arrcon:
    epsilon = 0.04*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(img, [approx], 0, (0), 3)
    x,y = approx[0][0]

    if len(approx) <=5:
        arrconchoosen.append(cnt)


#Get the ratio of the contours
arrcontchoosenlesa=[]
arr=[]
arrx=[]
arry=[]
i=0
for n, contour in enumerate(arrconchoosen):

    j=0
    for j in range (0, len(arrconchoosen[i])):
        arry.append(arrconchoosen[i][j][0][0])
        arrx.append(arrconchoosen[i][j][0][1])
        j=j+1
    minx=min(arrx)
    maxx= max(arrx)
    miny=min(arry)
    maxy= max(arry)
    length = (maxy -miny)
    width = (maxx -minx)
    ratio = length / width
    #print(maxy)
    #print(miny)
    #print(maxx)
    #print(minx)
    #print(ratio)
    if (ratio >= 1.5 and ratio <= 3.2):
        arrcontchoosenlesa.append(contour)
        arr.append(minx)
        arr.append(maxx)
        arr.append(miny)
        arr.append(maxy)
    arry.clear()
    arrx.clear()
    i=i+1


#Get the area of the rectangle contours
arrlast=[]
i=0
arry=[]
arrx=[]
for n, contour in enumerate(arrcontchoosenlesa):

    j=0
    for j in range (0, len(arrcontchoosenlesa[i])):
        arry.append(arrcontchoosenlesa[i][j][0][0])
        arrx.append(arrcontchoosenlesa[i][j][0][1])
        j=j+1
    minx=min(arrx)
    maxx= max(arrx)
    miny=min(arry)
    maxy= max(arry)
    length = (maxy -miny)
    width = (maxx -minx)
    # print(maxy)
    #print(miny)
    #print(maxx)
    #print(minx)
    #print(length)
    #print(width)
    area = abs (length *width)
    #print (area)
    if ( area >=3000 ):
       arrlast.append(contour)

    arry.clear()
    arrx.clear()
    i=i+1
for c in arrlast:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)

    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi.append([x, y , x+w, y+h])

#print (roi)

roii=roi[0]

#plt.imshow(img)
#plt.imshow(Originalrgb[roii[1]:roii[3], roii[0]:roii[2]])


imagegg=Originalrgb[roii[1]:roii[3], roii[0]:roii[2]]


cv2.imwrite('messigray.png',imagegg)


hya  =io.imread('messigray.png')

def compute_skew(file_name):

    #load in grayscale:
    src = cv2.imread(file_name,0)
    height, width = src.shape[0:2]

    #invert the colors of our image:
    cv2.bitwise_not(src, src)

    #Hough transform:
    minLineLength = width/2.0
    maxLineGap = 20
    lines = cv2.HoughLinesP(src,1,np.pi/180,100,minLineLength,maxLineGap)

    #calculate the angle between each line and the horizontal line:
    angle = 0.0
    nb_lines = len(lines)


    for line in lines:
        angle += math.atan2(line[0][3]*1.0 - line[0][1]*1.0,line[0][2]*1.0 - line[0][0]*1.0);

    angle /= nb_lines*1.0

    return angle* 180.0 / np.pi

anglee = compute_skew('messigray.png')
#print (anglee)



def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
ro = rotateImage(imagegg, -1* anglee)

show_images([ro])

print(timer()-start,"time end")
