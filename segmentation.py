from commonfunctions import *
from skimage.morphology import binary_erosion, binary_dilation, binary_closing,skeletonize, thin
from skimage.measure import find_contours
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt
from scipy.signal import find_peaks


@jit(nopython=True)
def adabtiveThreshold(inIm ,outIm,w,h):
    intImage = np.zeros((inIm.shape))
    index = 0
    count = 0
    t = 15
    s = int(w/8)
    s2 = int(s/2)
    #intImage = inIm.cumsum(axis=0).cumsum(axis=1)
    for i in range(h):
        sum = 0
        for j in range(w):
            sum = sum + inIm[i][j]
            if i == 0:
                intImage[i][j] = sum
            else:
                intImage[i][j] = intImage[i-1][j] + sum

    for i in range(w):
        for j in range(h):
            x1 = i - s2
            x2 = i + s2
            y1 = j - s2
            y2 = j + s2

            if x1 < 0:
                x1 = 0
            if x2 >= w:
                x2 = w - 1
            if y1 < 0:
                y1 = 0
            if y2 >= h:
                y2 = h - 1

            count = (x2-x1)*(y2-y1)
            sum = intImage[y2][x2] - intImage[y1-1][x2] - intImage[y2][x1-1] + intImage[y1-1][x1-1]

            if inIm[j][i]*count <= sum*(100-t)/100 :
                outIm[j][i] = 255
            else:
                outIm[j][i] = 0



@jit(nopython=True)
def verticalProjection(img):
    h, w = img.shape
    sumCols = []
    for j in range(w):
        col = img[0:h, j:j+1]
        sumCols.append(int(np.sum(col)/255))
    return sumCols


@jit(nopython=True)
def horizontalProjection(img):
    h, w = img.shape
    sumRows = []
    for j in range(h):
        row = img[j:j+1, 0:w]
        sumRows.append(int(np.sum(row)/255))
    return sumRows


def slice_digits(img):
    y = verticalProjection(img)
    xfrom = []
    xto = []
    xmin = -1
    xmax = -1
    miin = np.min(y)
    for i in range(len(y)):
        if y[i] > 15 and xmin == -1:
            xmin = i
            xfrom.append(xmin)
        elif y[i] < 15 and xmax == -1 and xmin != -1:
            xmin = -1
            xfrom.pop(len(xfrom)-1)
        elif y[i] > 15 and xmin != -1:
            xmax = i
        elif y[i] < 15 and xmax != -1:
            xto.append(xmax)
            xmin = -1
            xmax = -1


    newXfrom = []
    newXto = []
    for i in range(len(xto)):
        if (xto[i] - xfrom[i]) > 25:
            newXto.append(xto[i] + 10)
            if xfrom[i] >= 10:
                newXfrom.append(xfrom[i] - 10)
            else:
                newXfrom.append(0)


    charecters = []
    for i in range(len(newXto)):
        if len(newXto) > 0:
            charecters.append(img[:,newXfrom[i]:newXto[i]])
    if len(charecters) > 0:
        #show_images([charecters])
        return charecters




def process(image):
    #show_images([image])
    image  = rgb2gray(image)

    if np.max(image) <= 1:
        image = image*255
    image = image.astype('uint8')

    y,x = image.shape

    newX = x + int(x/16)
    newY = y + int(x/16)
    tempIm = np.zeros([newY,newX], dtype='uint8')
    tempIm[int(x/16):newY, int(x/16):newX] = np.copy(image)
#################################################################
    outIm = np.zeros([newY,newX], dtype='uint8')
    adabtiveThreshold(tempIm, outIm, newX, newY)

    newIm = np.copy(outIm[int(x/16):newY, int(x/16):newX])
    newY, newX = newIm.shape
###############################################################
    #show_images([newIm])
    wind = np.ones(shape=(7,7))
    im = median(newIm)

    im2 = binary_dilation(im,wind)


    im2 = binary_erosion(im2,wind)
    im2 = im2.astype('uint8')
    im2 = im2*255

    #show_images([im2])

    z = horizontalProjection(im2[int(0.15*im2.shape[0]):int(0.6*im2.shape[0]),:])

    cut = z.index(np.max(z))
    cut = cut + int(0.15*im2.shape[0])


    segIm = np.copy(im2[cut:im.shape[0],:])
    segY, segX = segIm.shape

    #show_images([segIm])


    #========================= Removing borders using vertical and horizontal projections ===============
    borders = horizontalProjection(segIm[int(0.9*segY):segY,:])
    for i in range(len(borders)):
        if borders[i] >= int(0.2*segX):
            segIm[int(0.9*segY)+i,:] = 0

    borders = horizontalProjection(segIm[0:int(0.1*segY),:])
    for i in range(len(borders)):
        if borders[i] >= int(0.2*segX):
            segIm[i,:] = 0

    borders = verticalProjection(segIm[:,0:int(0.03*segX)])
    for i in range(len(borders)):
        if borders[i] >= int(0.2*segY):
            segIm[:,i] = 0

    borders = verticalProjection(segIm[:,int(0.97*segX):segX])
    for i in range(len(borders)):
        if borders[i] >= int(0.2*segY):
            segIm[:,int(0.97*segX)+i] = 0

    borders = verticalProjection(segIm[:,int(0.49*segX):int(0.52*segX)])
    for i in range(len(borders)):
        if borders[i] >= int(0.3*segY):
            segIm[:,int(0.49*segX)+i] = 0

    #([segIm])
    #==================================================================================================

    segImNumbers = np.copy(segIm[0:segY,0:int(0.5*segX)])
    segImLetters = np.copy(segIm[0:segY,int(0.5*segX):segX])

    segYN, segXN = segImNumbers.shape
    segYL, segXL = segImLetters.shape

    xnums = slice_digits(segImNumbers)
    xletrs =slice_digits(segImLetters)


    return xnums,xletrs
