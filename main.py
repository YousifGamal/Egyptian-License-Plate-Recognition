from commonfunctions import *
from featureExtraction import *
from train import *
from Detection import *
from segmentation import *
from timeit import default_timer as timer
import concurrent.futures


letters,lettersDCT = readFeaturesFromFile('letters.txt',151)
numbers,numbersDCT = readFeaturesFromFile('numbers.txt',151)

letters = np.array(letters)
numbers = np.array(numbers)
i = 0
#paths = getImagesIn('Plates/Plates/one miss/2.jpg')
paths = ['Plates/Plates/working/15.jpg']
temp = []
totalOutput = []
totalOutputLens = []
totalOutputLetters = []
totalOutputLensLetters = []
tempLetters = []

def getSegmentedChars(img_paths):
    img = io.imread(img_paths)

    nums,letrs = process(img)

    temp = []
    tempLetters = []
    for n in nums:
        temp.append(detect_numbers(n,numbers,numbersDCT))
    for l in letrs:
        tempLetters.append(detect_letters(l,letters,lettersDCT))
    return temp,tempLetters,img_paths


with concurrent.futures.ThreadPoolExecutor() as executer:
    results =  executer.map(getSegmentedChars,paths)
    for result in results:
        print(numbers[result[0]],letters[result[1]],result[2])
