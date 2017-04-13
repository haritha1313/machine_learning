#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 12:12:38 2017

@author: pegasus
"""

import pandas as pd
import urllib2
import math


iris=pd.read_csv("Iris.csv")

iris.loc[iris["Species"] == "Iris-setosa", "NewSpecies"] = 0
iris.loc[iris["Species"] == "Iris-versicolor", "NewSpecies"] = 1
iris.loc[iris["Species"] == "Iris-virginica", "NewSpecies"] = 2

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

predictors=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg,iris[predictors],iris["NewSpecies"],cv=3)

alg.fit(iris[predictors],iris["NewSpecies"])
predictions=alg.predict(iris_test[predictors])
print(predictions)
