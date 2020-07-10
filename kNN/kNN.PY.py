from numpy import *
import operator

# Create the kNN classifier
def classify0(inX, dataSet, labels, k):
    # 1. Calculate he distance
    dataSetSize = dataSet.shape[0]
    temp = tile(inX, (dataSetSize, 1))
    subtract = temp - dataSet
    square = subtract ** 2
    summing = square.sum(axis = 1)
    distance = summing ** 0.5
    sortDistanceIndex = distance.argsort()
    # 2. Voting for the k nearest neighbors
    neighbors = { }
    for i in range(k):
        label = labels[sortDistanceIndex[i]]
        neighbors[label] = neighbors.get(label, 0) + 1
    # 3. Find the most frequent label
    sortLabels = sorted(neighbors.items(), key = operator.itemgetter(1), reverse = True)
    return sortLabels[0][0]

# The first simple example
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# The second example: dating recommendation application
# Prepared
# 1. Read data from the file
def file2Matrix(fileName):
    fr = open(fileName, "r")
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(fileName)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index, :] = listFromLine[0: 3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# 2. Normalizing data
# Normalization formula : newValue = (oldValue - min) / (max - min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# Testing
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2Matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))

# Using the system
def classifyPerson():
    resultList = ["not at all", "in small doses", "in large doses"]
    datingDataMat, datingLabels = file2Matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)

    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    userInputArr = [ffMiles, percentTats, iceCream]
    normUserInputArr = (userInputArr - minVals) / ranges
    classifierResult = classify0(normUserInputArr, normMat, datingLabels, 4)
    print("You will probably like this person " + resultList[classifierResult - 1])

classifyPerson()


