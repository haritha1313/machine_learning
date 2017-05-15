# -*- coding: utf-8 -*-
"""
Created on Sun May 14 17:58:30 2017

@author: pegasus
"""
import glob
import pandas as pd
from PIL import Image
import cv2
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

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
    pix=pix/255
    return pix

train = glob.glob('../input/train/**/*.jpg')+glob.glob('../input/additional/**/*.jpg')
train = pd.DataFrame([[p.split('/')[3], p.split('/')[4], p] for p in train], columns=['type','image','path'])[::5]
train = get_size(train)
train = train[train['size']!= '0 0'].reset_index(drop=True)
train_data = normalize(train['path'])
np.save('train.npy', train_data, fix_imports=True)

le=LabelEncoder()
train_target = le.fit_transform(train['type'].values)
print(le.classes_)
np.save('train_target.npy', train_target, fix_imports=True)

test = glob.glob('../input/test/*.jpg')
test = pd.DataFrame([[p.split('/')[3],p] for p in test], columns = ['image','path'])[::3]
test_data = normalize(test['path'])
np.save('test.npy', test_data, fix_imports=True)

test_id=test.image.values
np.save('test_id.npy', test_id, fix_imports=True)

train_data = np.load('train.npy')
train_target = np.load('train_target.npy')
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D,  MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

x_train,x_val_train,y_train,y_val_train = train_test_split(train_data,train_target,test_size=0.4, random_state=17)

def create_model(opt_='adamax'):
    model = Sequential()
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th', input_shape=(3, 32, 32))) #use input_shape=(3, 64, 64)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(12, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=opt_, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model

datagen = ImageDataGenerator(rotation_range=0.3, zoom_range=0.3)
datagen.fit(train_data)

model = create_model()
model.fit_generator(datagen.flow(x_train,y_train, batch_size=15, shuffle=True), nb_epoch=200, samples_per_epoch=len(x_train), verbose=20, validation_data=(x_val_train, y_val_train))

test_data = np.load('test.npy')
test_id = np.load('test_id.npy')

pred = model.predict_proba(test_data)
sub = pd.DataFrame(pred, columns=['Type_1','Type_2','Type_3'])
sub['image_name'] = test_id
sub.to_csv('submission.csv', index=False)
