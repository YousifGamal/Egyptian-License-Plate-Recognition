from commonfunctions import *
from featureExtraction import *
from train import *
from Detection import *
from segmentation import *
from timeit import default_timer as timer

letters,lettersDCT = readFeaturesFromFile('letters.txt')
numbers,numbersDCT = readFeaturesFromFile('numbers.txt')

'''
files = getImagesIn('Data_Set/toTest/')
for f in files:
    img  = io.imread(f)
    detect_letters(img,letters,lettersDCT)
'''
i = 0
paths = getImagesIn('New set/Frame1/')
tempn = []
totalOutputn = []
totalOutputLensn = []
templ = []
totalOutputl = []
totalOutputLensl = []

start1 = timer()
i = 0
while(i<13):
    img = io.imread(paths[i])
    #start1 = time.time()
    nums , lets = process(img)


    #end1 = time.time()
    #print("Elapsed in process" , (end1 - start1))

    #print(nums)
    tempn = []
    templ = []
    #start1 = time.time()
    for n in nums:
        tempn.append(detect_numbers(n,numbers,numbersDCT))

    for l in lets:
        templ.append(detect_letters(l,letters,lettersDCT))

    #end1 = time.time()
    #print("Elapsed in Dct Extraction" , (end1 - start1))

    '''
    print("start wait")
    time.sleep(1)
    print("end of wait")
    '''
    #print(temp)
    totalOutputn.append(tempn)
    totalOutputLensn.append(len(tempn))

    totalOutputl.append(templ)
    totalOutputLensl.append(len(templ))

    i +=1

#print(totalOutput)
outputLengthn = stats.mode(totalOutputLensn)[0][0]
newTotaloutputsn = []
for i in  range(len(totalOutputn)):
    if  (len(totalOutputn[i]) == outputLengthn):
        newTotaloutputsn.append(totalOutputn[i])
votesn = np.stack(newTotaloutputsn)
#print(votes)
for i in range(outputLengthn):
    #print("/////////////////////////////////////////////////////////////////////////")
    print(numbers[stats.mode(votesn[:,i])[0][0]])


outputLengthl = stats.mode(totalOutputLensl)[0][0]
newTotaloutputsl = []
for i in  range(len(totalOutputl)):
    if  (len(totalOutputl[i]) == outputLengthl):
        newTotaloutputsl.append(totalOutputl[i])
votesl = np.stack(newTotaloutputsl)
print(votesl)
for i in range(outputLengthl):
    #print("/////////////////////////////////////////////////////////////////////////")
    print(letters[stats.mode(votesl[:,i])[0][0]])

print(timer()-start1," time end ")









#print("frame endedddddddddddddddddddddd")



i = 0
paths = getImagesIn('New set/Frame2/')
tempn = []
totalOutputn = []
totalOutputLensn = []
templ = []
totalOutputl = []
totalOutputLensl = []

start1 = timer()
i = 0
while(i<13):
    img = io.imread(paths[i])
    #start1 = time.time()
    nums , lets = process(img)
    #end1 = time.time()
    #print("Elapsed in process" , (end1 - start1))

    #print(nums)
    tempn = []
    templ = []
    #start1 = time.time()
    for n in nums:
        tempn.append(detect_numbers(n,numbers,numbersDCT))

    for l in lets:
        templ.append(detect_letters(l,letters,lettersDCT))

    #end1 = time.time()
    #print("Elapsed in Dct Extraction" , (end1 - start1))

    '''
    print("start wait")
    time.sleep(1)
    print("end of wait")
    '''
    #print(temp)
    totalOutputn.append(tempn)
    totalOutputLensn.append(len(tempn))

    totalOutputl.append(templ)
    totalOutputLensl.append(len(templ))

    i +=1

#print(totalOutput)
outputLengthn = stats.mode(totalOutputLensn)[0][0]
newTotaloutputsn = []
for i in  range(len(totalOutputn)):
    if  (len(totalOutputn[i]) == outputLengthn):
        newTotaloutputsn.append(totalOutputn[i])
votesn = np.stack(newTotaloutputsn)
#print(votes)
for i in range(outputLengthn):
    #print("/////////////////////////////////////////////////////////////////////////")
    print(numbers[stats.mode(votesn[:,i])[0][0]])


outputLengthl = stats.mode(totalOutputLensl)[0][0]
newTotaloutputsl = []
for i in  range(len(totalOutputl)):
    if  (len(totalOutputl[i]) == outputLengthl):
        newTotaloutputsl.append(totalOutputl[i])
votesl = np.stack(newTotaloutputsl)
#print(

for i in range(outputLengthl):
    #print("/////////////////////////////////////////////////////////////////////////")
    print(letters[stats.mode(votesl[:,i])[0][0]])

print(timer()-start1," time end ")
