#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 02:29:25 2017

@author: pegasus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from os.path import isfile, isdir 

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

trainLabelCounts=train['label'].value_counts(sort=False)
#print(trainLabelCounts)

def getImage(data, *args):
    if args:
        number=args[0]
        specifiedData = data[data['label'] == number].values
    else:
        specifiedData=data.values
    randomNumber=np.random.choice(len(specifiedData)-1,1)
    return specifiedData[randomNumber, :]
    
def plotNumber(imageData, imageSize):
    if imageData.shape[1] == np.prod(imageSize):
        image=imageData[0,:].reshape(imageSize)
    elif imageData.shape[1]>np.prod(imageSize):
        label=imageData[0,0]
        image=imageData[0,1:].reshape(imageSize)
        plt.title('number: {}'.format(label))
    plt.imshow(image)
    plt.savefig("output.png")

imageSize=(28,28)
chosenNumber=1
#plotNumber(getImage(test),imageSize)

trainData=train.values[:,1:]
trainLabel=train.values[:,0]
testData=test.values

def preprocessing(data):
    minV=0
    maxV=255
    data= (data-minV)/ (maxV-minV)
    return data
    
def one_hot_encoding(data, numberOfClass):
    from sklearn import preprocessing
    lb=preprocessing.LabelBinarizer()
    lb.fit(range(numberOfClass))
    return lb.transform(data)
    
processedTrainData=preprocessing(trainData)
processedTestData=preprocessing(testData)
one_hot_trainLabel=one_hot_encoding(trainLabel,10)

fileName='mnist.p'
if not isfile(fileName):
    pickle.dump((processedTrainData, trainLabel, one_hot_trainLabel, processedTestData),open(fileName, 'wb'))
    
fileName='mnist.p'
trainData,trainLabel,one_hot_trainLabel,testData = pickle.load(open(fileName, mode='rb'))

def getInputTensor(features, numberOfClass):
    inputT =tf.placeholder(dtype=tf.float32, shape= (None, features), name='input')
    labelT=tf.placeholder(dtype=tf.float32, shape =(None, numberOfClass), name='label')
    keep_prob=tf.placeholder(dtype=tf.float32)
    
    return inputT, labelT, keep_prob
    
def hiddenLayer(inputT, numberOfNodes):
    inputSize = inputT.get_shape().as_list()[1]
    weights=tf.Variable(tf.truncated_normal((inputSize, numberOfNodes)), dtype=tf.float32)
    biases=tf.zeros((numberOfNodes), dtype=tf.float32)
    
    hiddenNodes=tf.add(tf.matmul(inputT, weights, biases))
    hiddenOutput=tf.nn.sigmoid(hiddenNodes)
    return hiddenOutput
    
def outputLayer(hiddenOutput, numberOfClass):
    inputSize=hiddenOutput.get_shape().as_list()[1]
    weights=tf.Variable(tf.truncated_normal((inputSize, numberOfClass)), dtype=tf.float32)
    biases=tf.zeros((numberOfClass),dtype=tf.float32)
    
    output=tf.add(tf.matmul(hiddenOutput,weights),biases)
    return output
    
def build_nn(inputT, numberOfNodes, numberOfClass, keep_prob):
    fc1=hiddenLayer(inputT, numberOfNodes)
    output=outputLayer(fc1, numberOfClasses)
    return output
    
numberOfNodes=256
batchSize=128
numberOfEpoch=20
learningRate=0.01
keep_prob_rate=1.0

numberOfClass=10
imageSize=(28,28)

features=np.prod(imageSize)
graph=tf.Graph()
tf.reset_default_graph()
with graph.as_default():
    inputT, labelT, keep_prob =getInputTensor(features, numberOfClass)
    logits=build_nn(inputT, numberOfNodes, numberOfClass, keep_prob)
    probability=tf.nn.softmax(logits, name='probability')
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = labelT))
    optimizer=tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)
    
    correctPrediction=tf.equal(tf.argmax(probability,1),tf.argmax(labelT,1))
    accuracy=tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
    
from sklearn.model_selection import train_test_split

def printResult(epoch, numberOfEpoch, trainLoss, validationLoss, validationAccuracy):
    print("Epoch: {}/{}".format(epoch+1, numberOfEpoch), '\tTraining Loss: {:.3f}'.format(trainingLoss), '\tValidation Loss: {: .3f}'.format(validationLoss), '\tAccuracy: {; .2f}%'.format(validationAccuracy*100))
    
    save_dir='./save'
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(numberOfEpoch):
            train_x, val_x, train_y, val_y =train_test_split(trainData, one_hot_trainLabel,test_size=0.2)
            for i in range(0,len(train_x), batchSize):
                trainLoss, _, _ = sess.run([cost,probability, optimizer], feed_dict = {
                    inputT: train_x[i: i+batchSize],
                    labelT: train_y[i: i+batchSize],
                    keep_prob: keep_prob_rate
                    })
            valAcc, valLoss=sess.run([accuracy, cost], feed_dict = {
                inputT: val_x,
                label_T: val_y,
                keep_prob: 1.0
                })
                
            printResult(epoch, numberOfEpoch, trainLoss, valLoss, valAcc)
        saver=tf.train.Saver()
        saver.save(sess,save_dir)
                




