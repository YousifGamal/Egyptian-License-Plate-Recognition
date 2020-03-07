import cv2
import time
from commonfunctions import *
from featureExtraction import *
from train import *
from Detection import *
from segmentation import *
# Opens the Video file
"""cap= cv2.VideoCapture('s2.mp4')
i=0
frames = 10
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if frames == 0:
        frames = 10
        cv2.imwrite('frames/kang'+str(i)+'.jpg',frame)
    i+=1
    frames-=1


cap.release()
cv2.destroyAllWindows()
"""


img = rgb2gray(io.imread('Data_Set/Taaah/0.png'))
img = np.pad(img,((0,0),(8,8)),mode='constant',constant_values = (0))

letters,lettersDCT = readFeaturesFromFile('letters.txt')
numbers,numbersDCT = readFeaturesFromFile('numbers.txt')

print(letters[detect_letters(img,letters,lettersDCT)])

