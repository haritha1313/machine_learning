#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:53:28 2017

@author: pegasus
"""
import pandas
from sklearn.feature_selection import SelectKBest, f_classif 
import numpy as np
from sklearn import cross_validation
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LinearRegression

forest=pandas.read_csv("train.csv")


predictors=list(forest.axes[1])

selector = SelectKBest(f_classif, k=9)

selector.fit(forest[predictors], forest["Cover_Type"])

scores = -np.log10(selector.pvalues_)

submission = pandas.DataFrame({
        "Feature": forest.columns,
        "Value": scores
    })

jo=list(submission[submission["Value"]>60]["Feature"][1:])

from sklearn.ensemble import GradientBoostingClassifier

alg=GradientBoostingClassifier(random_state=1, n_estimators=20, max_depth=4)


kf=cross_validation.KFold(forest.shape[0], n_folds=3, random_state=1)

scores=cross_validation.cross_val_score(alg, forest[jo], forest["Cover_Type"], cv=kf)


forest_test=pandas.read_csv("test.csv")

alg.fit(forest[jo[:-1]],forest["Cover_Type"])

predictions=alg.predict(forest_test[jo[:-1]])

sub=pandas.DataFrame({ "Cover_Type": predictions,
                      "Id": forest_test["Id"]
                     
                     })
newhh=sub[['Id','Cover_Type']]

#print newhh
newhh.to_csv("submissionforest2.csv", index=False)