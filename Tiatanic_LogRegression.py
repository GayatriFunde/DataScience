# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 21:46:09 2019

@author: windows
"""

import os
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model,model_selection
import numpy as np

os.getcwd()
os.chdir('E:\\Python_Titanic')
titanic_train = pd.read_csv('train.csv')
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train,columns=['Pclass','Sex','Embarked'])
titanic_train1.shape
titanic_train1.info()

x_train = titanic_train1.drop(['PassengerId','Survived','Name','Age','Ticket','Cabin'],1)
x_train.info()
y_train = titanic_train1['Survived']

lr_estimator= linear_model.LogisticRegression(random_state=500)
lr_Grid = {'C':list(np.arange(0.1,1.0,0.19)),'penalty':['l1','l2'],'max_iter':list(range(20,51,10))}
lr_grid_estimator = model_selection.GridSearchCV(lr_estimator,lr_Grid,cv=10,n_jobs=1,verbose=3)
lr_grid_estimator.fit(x_train,y_train)
lr_grid_estimator.cv_results_
final_model = lr_grid_estimator.best_estimator_
lr_grid_estimator.best_score_
lr_grid_estimator.best_params_

final_model.coef_
final_model.intercept_

titanic_test = pd.read_csv('test.csv')
titanic_test.shape
titanic_test1= pd.get_dummies(titanic_test,columns=['Pclass','Sex','Embarked'])
titanic_test1.shape
x_test = titanic_test1.drop(['PassengerId','Name','Age','Ticket','Cabin'],1)
x_test.shape
x_test.info()

mean_imputer = preprocessing.Imputer()
mean_imputer.fit(x_test[['Fare']])
x_test['Fare'] = mean_imputer.transform(x_test[['Fare']])
titanic_test['Survived'] = final_model.predict(x_test)

titanic_test.to_csv('Submission_LogRreg.csv', columns=['PassengerId','Survived'], index=False)


