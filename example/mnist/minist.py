# coding=utf-8

import cPickle, gzip, os, sys
import numpy as np
from deep8 import *

def loadData(dataPath):
    # Load the dataset
    f = gzip.open(dataPath, 'rb')
    trainSet, validSet, testSet = cPickle.load(f)
    f.close()

    return (trainSet[0], trainSet[1], validSet[0], validSet[1], testSet[0], testSet[1])

# load data
trainX, trainY, validX, validY, testX, testY = loadData(os.getcwd() + "/data/mnist.pkl.gz")

'''
trainX [50000, 784]
trainY [50000, ]
validX [10000, 784]
validY [10000, ]
testX [10000, 784]
testY [10000, ]
'''

trainer  = AdagradTrainer(learningRate = 0.01)
executor = EagerExecutor(tr=trainer)

x = inputParameter(executor, [784])
y = inputParameter(executor, [10])

w = parameter(executor, [10, 784])
b = parameter(executor, [10])

for i in range(len(trainX)):
    one_hot_y = np.zeros([10], dtype=np.float32)
    one_hot_y[trainY[i]] = 1.0

    x.feed(trainX[i])
    y.feed(one_hot_y)

    loss = (w * x + b).softmax().crossEntropy(y)

    print i, ", loss => ", loss.valueString()

    loss.backward()

pred = np.zeros([10], dtype=np.float32)

correct = 0
wrong = 0

for i in range(len(testX)):
    x.feed(testX[i])

    ret = (w * x + b).softmax()

    ret.fetch(pred)

    executor.clearIntermediaryNodes()

    if np.argmax(pred) == testY[i]:
        correct += 1
    else:
        wrong += 1

print "Total:", correct + wrong, ", Correct:", correct, ", Wrong:", wrong, "Accuracy:", (1.0 * correct) / (correct + wrong)






