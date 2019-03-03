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

executor     = EagerExecutor()
learningRate = ConstantLearningRateIterator(0.01)
trainer      = AdagradTrainer(learningRate = learningRate)

x = parameter(executor, [784], False)
y = parameter(executor, [10], False)

w = parameter(executor, [10, 784])
b = parameter(executor, [10])

for i in range(len(trainX)):
    one_hot_y = np.zeros([10], dtype=np.float32)
    one_hot_y[trainY[i]] = 1.0

    x.feed(trainX[i])
    y.feed(one_hot_y)

    loss = (w * x + b).softmax().crossEntropy(y)

    print i + 1, ", loss => ", loss.valueStr()

    loss.backward()

    trainer.train(executor)

pred = np.zeros([10], dtype=np.float32)

correct = 0
wrong = 0

for i in range(len(testX)):
    x.feed(testX[i])

    ret = (w * x + b).relu().softmax()

    ret.fetch(pred)

    executor.clearInterimNodes()

    if np.argmax(pred) == testY[i]:
        correct += 1
    else:
        wrong += 1

print "Total:", correct + wrong, ", Correct:", correct, ", Wrong:", wrong, "Accuracy:", (1.0 * correct) / (correct + wrong)


