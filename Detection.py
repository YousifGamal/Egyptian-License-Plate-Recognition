from featureExtraction import *

def detect_letters(image , letters , lettersDCT):
    n = 100
    test = extractFeatures(image , n)
    min_index = -1
    ecos = np.zeros(len(letters))

    for i in range (len(ecos)):
        for j in range (n):
            ecos[i] +=  (lettersDCT[i][j]-test[j])**2
        ecos[i]  = np.sqrt(ecos[i])


    min_index = np.argmin(ecos)
    if min_index == 4 or min_index == 9 or min_index == 14:
        min_index = roadFork(image,n = 150)

    return min_index




def detect_numbers(image , numbers , numbersDCT):
    n = 150
    test = extractFeatures(image , n,0)
    min_index = -1
    ecos = np.zeros(len(numbers))

    for i in range (len(ecos)):
        for j in range (n):
            ecos[i] +=  (numbersDCT[i][j]-test[j])**2
        ecos[i]  = np.sqrt(ecos[i])


    min_index = np.argmin(ecos)
    return min_index
