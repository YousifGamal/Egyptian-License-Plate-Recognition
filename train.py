from featureExtraction import *


def writeToFile(fileName,labels,featureVectors):
    file=open(fileName,"w+")
    for i in range(len(labels)):
        file.write(labels[i]+"\n")
        for j in featureVectors[i]:
            file.write("%d\n"%j)
    file.close()

def readFeaturesFromFile(fileName,featureVectorSize = 151):
    f = open(fileName,'rt')
    labels = []
    featureVectors = []
    temp = []
    i = 0
    for line in f:
        if (i%(featureVectorSize+1) == 0):
            labels.append(str(line.rstrip()))
            if (i > 0):
                featureVectors.append(temp)
            temp = []
        else:
            temp.append(float(line))
        i +=1
    featureVectors.append(temp)
    return labels,featureVectors

def trainLetters():
    letters = ['3en','alf','bih','dal'
               ,'feh','gem','heh','lam'
               ,'non','qaf','reh','sad'
               ,'sen','mem','wow','yeh',
               'tah']
    lettersDCT = []
    for l in letters:
        lettersDCT.append(trainFeatures(getImagesIn('Data_Set/Letters/'+l),150))
    writeToFile('letters.txt',letters,lettersDCT)

def trainNumbers():
    numbers = ['1','2','3','4','5','6','7','8','9']
    numbersDCT = []
    for n in  numbers:
        numbersDCT.append(trainFeatures(getImagesIn('Data_Set/Numbers/'+n),150))
    writeToFile('numbers.txt',numbers,numbersDCT)

#trainLetters()
#trainNumbers()


        


