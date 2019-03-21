# -*- coding: utf-8 -*-
"""
Created on Sun May 14 17:58:30 2017

@author: pegasus
"""
import glob
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

def calc_size(path):
    try:
        img = Image.open(path)
        return [path, {'size':img.size}]
    except:
        print(path)
        return [path, {'size':[0,0]}]

def get_size(images):
    collect={}
    lo = Pool(cpu_count())
    ans = lo.map(calc_size, images['path'])
    for i in range(len(ans)):
        collect[ans[i][0]]=ans[i][1]
    images['size']=images['path'].map(lambda x: ' '.join(str(s) for s in collect[x]['size']))
    return images
    
def get_data(path1):
    img1 = cv2.imread(path1)
    resu = cv2.resize(img1, (32,32), cv2.INTER_LINEAR)
    return [path1, resu]    
    
def normalize(paths):
    coll={}
    lo=Pool(cpu_count())
    ans = lo.map(get_data, paths)
    for i in range(len(ans)):
        coll[ans[i][0]]=ans[i][1]
    ans=[]
    pix = [coll[a] for a in paths]
    pix=np.array(pix, dtype=np.uint8)
    pix = pix.astype('float32')
    pix=pix/255.0
    print(pix.shape)
    return pix

train = glob.glob('../input/train/**/*.jpg')+glob.glob('../input/additional/**/*.jpg')
train = pd.DataFrame([[p.split('/')[3], p.split('/')[4], p] for p in train], columns=['type','image','path'])
train = get_size(train)
train = train[train['size']!= '0 0'].reset_index(drop=True)
train_data = normalize(train['path'])

le=LabelEncoder()
label = le.fit_transform(train['type'].values)

from keras.utils import np_utils
print(le.classes_)

test = glob.glob('../input/test/*.jpg')
test = pd.DataFrame([[p.split('/')[3],p] for p in test], columns = ['image','path'])
test_data = normalize(test['path'])

test_id=test.image.values

from sklearn.model_selection import train_test_split

(trainData, testData, trainLabels, testLabels) = train_test_split(train_data, label, test_size=0.25, random_state=42)

model = Sequential()
model.add(Convolution2D(32, 3,3, activation='relu', input_shape=(32,32,3)))
print(model.output_shape)

model.add(Convolution2D(32, 3, 3, activation='relu'))
print(model.output_shape)

model.add(MaxPooling2D(pool_size=(2,2)))
print(model.output_shape)

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(768,init="uniform", activation='relu'))
print(model.output_shape)

model.add(Dropout(0.5))
model.add(Dense(384, init="uniform", activation="relu"))
print(model.output_shape)

model.add(Dense(3))
model.add(Activation("softmax"))


trainLabels = trainLabels.reshape((-1, 1))

sgd = SGD(lr=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(trainData, trainLabels, nb_epoch=50, batch_size=128, verbose=1)

(loss, accuracy) = model.evaluate(testData, testLabels,	batch_size=128, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
pred = model.predict_proba(test_data)
df = pd.DataFrame(pred, columns=['Type_1','Type_2','Type_3'])
df['image_name'] = test_id
newhh=df[['image_name','Type_1','Type_2','Type_3']]
newhh.to_csv('submission.csv', index=False)
