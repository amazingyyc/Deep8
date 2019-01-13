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

trainer  = AdagradTrainer(learningRate = 0.0001)
executor = EagerExecutor(tr=trainer)

x = inputParameter(executor, [28, 28, 1])
y = inputParameter(executor, [10])

# first convolution
w_conv1 = parameter(executor, [32, 5, 5, 1])
b_conv1 = parameter(executor, [32])

# second convolution
w_conv2 = parameter(executor, [64, 5, 5, 32])
b_conv2 = parameter(executor, [64])

# full connected layer
w_fc1 = parameter(executor, [1024, 4 * 4 * 64])
b_fc1 = parameter(executor, [1024])

# full connected layer
w_fc2 = parameter(executor, [10, 1024])
b_fc2 = parameter(executor, [10])

epoch = 10

for e in range(epoch):
    for i in range(len(trainX)):
        one_hot_y = np.zeros([10], dtype=np.float32)
        one_hot_y[trainY[i]] = 1.0

        x.feed(trainX[i])
        y.feed(one_hot_y)

        layer1 = (x.conv2d(w_conv1) + b_conv1).relu().maxPooling2d(filterHeight=2, filterWidth=2, strideY=2, strideX=2)
        layer2 = (layer1.conv2d(w_conv2) + b_conv2).relu().maxPooling2d(filterHeight=2, filterWidth=2, strideY=2, strideX=2)
        layer3 = (w_fc1 * layer2.reShape([4 * 4 * 64]) + b_fc1).relu()
        layer4 = w_fc2 * layer3 + b_fc2

        loss = layer4.softmax().crossEntropy(y)

        print "epoch:", e, ", step:", i, ", loss => ", loss.valueString()

        loss.backward()

pred = np.zeros([10], dtype=np.float32)

correct = 0
wrong   = 0

for i in range(len(testX)):
    x.feed(testX[i])

    layer1 = (x.conv2d(w_conv1) + b_conv1).relu().maxPooling2d(filterHeight=2, filterWidth=2, strideY=2, strideX=2)
    layer2 = (layer1.conv2d(w_conv2) + b_conv2).relu().maxPooling2d(filterHeight=2, filterWidth=2, strideY=2, strideX=2)
    layer3 = (w_fc1 * layer2.reShape([4 * 4 * 64]) + b_fc1).relu()
    layer4 = w_fc2 * layer3 + b_fc2
    ret    = layer4.softmax()

    ret.fetch(pred)

    executor.clearIntermediaryNodes()

    if np.argmax(pred) == testY[i]:
        correct += 1
        print "test ", i, " => right"
    else:
        wrong += 1
        print "test ", i, " => wrong"

print "Total:", correct + wrong, ", Correct:", correct, ", Wrong:", wrong, "Accuracy:", (1.0 * correct) / (correct + wrong)















