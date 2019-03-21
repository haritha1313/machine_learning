#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:02:06 2017

@author: pegasus
"""


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import validation_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation 
train=pd.read_csv("train.csv")
#test=pd.read_csv("test.csv")

#ids=test["Id"]
#test=test.drop("Id",axis=1)
train=train.drop("Id",axis=1)

y=train["Cover_Type"]
xtrain=train.drop("Cover_Type",axis=1)

predictors=list(xtrain.columns)
print predictors

alg=RandomForestClassifier(random_state=0,max_features='sqrt')

def feat_eng(df):
    
    #absolute distance to water
    df['Distance_To_Hydrology']=(df['Vertical_Distance_To_Hydrology']**2.0+
                                 df['Horizontal_Distance_To_Hydrology']**2.0)**0.5
    
feat_eng(xtrain)
#feat_eng(test)


"""
prange=np.arange(50,650,50)
train_scores, cv_scores = validation_curve(alg, xtrain, y,param_name='n_estimators',
                                              param_range=prange)


plt.xlabel('n_estimators')
plt.ylabel('Mean CV score')
plt.plot(prange, np.mean(cv_scores, axis=1), label="Cross-validation score",color="g")
"""

alg.set_params(n_estimators=450)

alg.fit(xtrain,y)
print pd.DataFrame(alg.feature_importances_,index=xtrain.columns).sort([0], ascending=False)
"""
score=cross_validation.cross_val_score(alg,xtrain,y).mean()
print score"""
