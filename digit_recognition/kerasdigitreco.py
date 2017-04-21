#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 22:49:40 2017

@author: pegasus
"""

import theano
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

trainvals=train.values[:,1:]
trainlabels=train.values[:,0]

testvals=test.values

trainvals=trainvals.reshape(trainvals.shape[0],28,28,1)
testvals=testvals.reshape(testvals.shape[0],28,28,1)

trainvals=trainvals.astype('float32')
testvals=testvals.astype('float32')

trainvals /=255
testvals /=255

trainlabels=np_utils.to_categorical(trainlabels, 10)

model=Sequential()

model.add(Convolution2D(32,3,3, activation='relu', input_shape=(28,28,1)))

model.add(Convolution2D(32,3,3,activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(trainvals,trainlabels,batch_size=32,nb_epoch=10,verbose=0)

prediction=model.predict(testvals, batch_size=32, verbose=0)

pd.DataFrame({'ImageId': list(range(len(predictions)+1)), 'Label': predictions}).to_csv('predictions.csv',index=False,header=True)
