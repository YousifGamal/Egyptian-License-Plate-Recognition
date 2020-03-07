from commonfunctions import *
from featureExtraction import *
from train import *
from Detection import *
from segmentation import *
from timeit import default_timer as timer
import concurrent.futures

letters,lettersDCT = readFeaturesFromFile('letters.txt',151)
numbers,numbersDCT = readFeaturesFromFile('numbers.txt',151)

i = 0
paths = getImagesIn('New set/Frame1/')
temp = []
totalOutput = []
totalOutputLens = []
totalOutputLetters = []
totalOutputLensLetters = []
tempLetters = []

def voting(flag):
    if flag == 0:
        totalOutputs = totalOutput
        totalOutputLenses = totalOutputLens
    else:
        totalOutputs = totalOutputLetters
        totalOutputLenses = totalOutputLensLetters
    outputLength = stats.mode(totalOutputLenses)[0][0]
    newTotaloutputs = []
    for i in  range(len(totalOutputs)):
        if  (len(totalOutputs[i]) == outputLength):
            newTotaloutputs.append(totalOutputs[i])
    votes = np.stack(newTotaloutputs)
    #print(votes)

    VotingResult =[]
    for i in range(outputLength):
        if flag == 0:
            VotingResult.append(numbers[stats.mode(votes[:,i])[0][0]])
        else:
            VotingResult.append(letters[stats.mode(votes[:,i])[0][0]])
    return VotingResult

def getSegmentedChars(img_paths):
    img = io.imread(img_paths)
    nums,letrs = process(img)
    temp = []
    tempLetters = []
    for n in nums:

        temp.append(detect_numbers(n,numbers,numbersDCT))
    for l in letrs:

        tempLetters.append(detect_letters(l,letters,lettersDCT))
    return temp,tempLetters


#start = timer()

with concurrent.futures.ThreadPoolExecutor() as executer:
    results =  executer.map(getSegmentedChars,paths)
    for result in results:
        totalOutput.append(result[0])
        totalOutputLens.append(len(result[0]))
        totalOutputLetters.append(result[1])
        totalOutputLensLetters.append(len(result[1]))

with concurrent.futures.ThreadPoolExecutor() as executerVoting:
    res = executerVoting.map(voting,[0,1])
    for r in res:
        print(r)

#print(timer()-start," time end ")
