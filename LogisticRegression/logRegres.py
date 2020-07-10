from numpy import *

# Data Set and Label Set
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

# The sigmoid function
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

# Train
# Get the coefficient of the regression formula
# normal gradient descent
def gradientAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

# stochastic gradient descent
def stoGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(dataMatrix[i] * weights)
        error = classLabels[i] - h
        weights = weights + alpha * dataMatrix[i] * error
    return weights

# best
def stoGradAscent1(dataMatrix, classLabels, numIter = 150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.1
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights

# Test
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def test():
    dataMat, labelMat = loadDataSet()
    # Train and get the weights
    weights = stoGradAscent1(array(dataMat), labelMat)
    x1 = float(input("Enter feature 1:"))
    x2 = float(input("Enter feature 2:"))
    inputVect = [1.0, x1, x2]
    result = classifyVector(inputVect, weights)
    print("The horse belongs to :", result)

test()


