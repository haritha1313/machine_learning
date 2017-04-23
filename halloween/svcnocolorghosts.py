# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelBinarizer
from sklearn import svm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
ids=train.values[:,0]
testid=test.values[:,0]
train=train.drop('id',axis=1)
lb=LabelBinarizer()
train=train.drop('color',axis=1)

new=train["type"]

test=test.drop(['color','id'],axis=1)
train=train.drop('type',axis=1)
X=pd.DataFrame.as_matrix(train)
y=pd.DataFrame.as_matrix(new)
alg = svm.SVC(kernel='linear', C = 1.0)
alg.fit(X,y)
prediction=alg.predict(test)
print(testid,prediction)
submission=pd.DataFrame({'id':testid, 'type': prediction})
submission.to_csv("submissionghost.csv",index=False)
# Any results you write to the current directory are saved as output.