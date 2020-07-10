import math
from numpy import *
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

# Shannon Entropy: bigger, messier, worse
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = { }
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1 :])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# find the best attribute to split the data set
# information gains = baseEntropy - newEntropy
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# Address the problem of running out of attributes but the class labels are not the same
# similar to kNN classifier in chapter 2
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

# Create the decision tree and train the tree
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # Two edge cases
    # stop when all classes are equal
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # when no more features, choose the majority
    if len(dataSet[0]) == -1:
        return majorityCnt(classList)

    # general cases
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}  # dictionary
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# Test the tree: build the classifier using decision tree
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# Serialize the tree object (dictionary) for later use using pickle
def storeTree(inputTree, fileName):
    import pickle
    fw = open(fileName, "w")
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(fileName):
    import pickle
    fr = open(fileName)
    return pickle.load(fr)

# Real World Example: lenses
def openFile(fileName):
    fr = open(fileName, "r")
    returnMat = [line.strip().split("\t") for line in fr.readlines()]
    classLabelVector = ['age', 'prescript', 'astigmatic', 'tearRate']
    return returnMat, classLabelVector

# Use
def main():
    originalData, labels = openFile("lenses.txt")
    anotherLabelList = labels.copy()
    tree = createTree(originalData, anotherLabelList)
    first = input("Young, pre, or presbyopic?").lower()
    second = input("Hyper or myope?").lower()
    third = input("Astigmatic? yes or no").lower()
    fourth = input("Reduced or normal?").lower()
    testVec = [first, second, third, fourth]
    classifyResult = classify(tree, labels, testVec)
    print("Lenses you should prescribe: " + classifyResult)

main()