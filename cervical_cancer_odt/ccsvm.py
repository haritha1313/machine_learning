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
    pix = [np.ravel(coll[a]) for a in paths]
    pix=np.array(pix, dtype=np.uint8)
    pix = pix.astype('float32')
    pix=pix/255
    return pix

train = glob.glob('../input/train/**/*.jpg')+glob.glob('../input/additional/**/*.jpg')
train = pd.DataFrame([[p.split('/')[3], p.split('/')[4], p] for p in train], columns=['type','image','path'])
train = get_size(train)
train = train[train['size']!= '0 0'].reset_index(drop=True)
train_data = normalize(train['path'])
np.save('train.npy', train_data, fix_imports=True)

le=LabelEncoder()
train_target = le.fit_transform(train['type'].values)
print(le.classes_)
np.save('train_target.npy', train_target, fix_imports=True)

test = glob.glob('../input/test/*.jpg')
test = pd.DataFrame([[p.split('/')[3],p] for p in test], columns = ['image','path'])
test_data = normalize(test['path'])
np.save('test.npy', test_data, fix_imports=True)

test_id=test.image.values
np.save('test_id.npy', test_id, fix_imports=True)


test_data = np.load('test.npy')
test_ids = np.load('test_id.npy')
train_data = np.load('train.npy')
train_target = np.load('train_target.npy')
print(train_data.shape)
from sklearn.model_selection import train_test_split
from sklearn import svm
x_train, x_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.4, random_state=0)

clf = svm.SVC(kernel='linear', C=1, probability=True).fit(train_data, train_target)
results = clf.predict_proba(test_data)

type1=results[:,0]
type2=results[:,1]
type3=results[:,2]
sub=pd.DataFrame({'image_name': test_ids, 'Type_1':type1, 'Type_2': type2, 'Type_3': type3})
newhh=sub[['image_name', 'Type_1', 'Type_2', 'Type_3']]
newhh.to_csv("svmcc.csv",index=False)
