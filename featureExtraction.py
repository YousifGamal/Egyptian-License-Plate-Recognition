from commonfunctions import *


@jit(nopython=True)
def viewPort(original,X,Y):

    Xwmax = original.shape[0]-1
    Ywmax = original.shape[1]-1
    Xvmax = X-1
    Yvmax = Y-1
    Sx = (Xvmax)/(Xwmax)
    Sy = (Yvmax)/(Ywmax)
    new  = np.zeros((Xvmax+1,Yvmax+1))
    count = np.zeros((Xvmax+1,Yvmax+1))

    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            newx = int((i)*Sx)
            newy = int((j)*Sy)
            new[newx,newy] += original[i,j]/255
            count[newx,newy] += 1

    for i in range (Xvmax+1):
        for j in range(Yvmax+1):
            if (count[i,j] == 0):
                new[i,j] = 255
            else:
                new[i,j] = int(round(new[i,j]/count[i,j])*255)

            if(new[i,j] > 0):
                new[i,j] = 255

    return new


def getDCT(image,n):
    image = np.float32(image)
    dst =  fftpack.dct(fftpack.dct( image, axis=0,type=2, norm='ortho' ),axis=1, type=2, norm='ortho')
    rows=dst.shape[0]
    columns=dst.shape[1]
    matrix = dst
    solution=[[] for i in range(rows+columns-1)]
    for i in range(rows):
        for j in range(columns):
            sum=i+j
            if(sum%2 ==0):

            #add at beginning
                solution[sum].insert(0,matrix[i][j])
            else:

                #add at end of the list
                solution[sum].append(matrix[i][j])
    sol = np.hstack(solution)
    return sol[0:n+1]



def trainFeatures(images=[],n=150):
    total = []
    for image in images:

        img = io.imread(image)

        img = viewPort(img,X=150,Y=50)
        fv = getDCT(img,n)
        total.append(fv)
    for i in range(1,len(total)):
        total[0] += total[i]
    total[0] = total[0]/len(total)
    return total[0]


def extractFeatures(img,n=100,flag = 1):
    if flag == 1:
        img = viewPort(img,X=150,Y=50)
    else:
        img = viewPort(img,X=75,Y=40)
    fv = getDCT(img,n)
    return fv

def roadFork(img,n = 150):

    img = viewPort(img,X = 80,Y = 120)
    wow = viewPort(rgb2gray(io.imread('Data_Set/Letters/wow/183.png')),X = 80,Y = 120)
    qaf = viewPort(rgb2gray(io.imread('Data_Set/Letters/qaf/81.png')),X = 80,Y = 120)
    feh = viewPort(rgb2gray(io.imread('Data_Set/Letters/feh/150.png')),X = 80,Y = 120)
    img = getDCT(img,n)
    wow = getDCT(wow,n)
    qaf = getDCT(qaf,n)
    feh = getDCT(feh,n)
    compares = [wow,qaf,feh]
    comparesIndices = [14,9,4]

    ecoss = np.zeros(len(compares))

    for i in range (len(ecoss)):
        for j in range (n):
            ecoss[i] +=  (compares[i][j]-img[j])**2
        ecoss[i]  = np.sqrt(ecoss[i])


    min_index = np.argmin(ecoss)
    if(np.abs(ecoss[0]-ecoss[1]) < 1000):
        return 9
    else:
        return comparesIndices[min_index]

def getImagesIn(path):
    images = []
    files = []
    for r,d,f in os.walk(path):
        for file in f:
            files.append(os.path.join(r,file))
    return files
