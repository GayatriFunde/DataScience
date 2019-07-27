# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 19:45:17 2019

@author: Gayatri
"""

import pandas as pd
from sklearn import tree
import os
import pydot,io
from sklearn import model_selection

os.chdir('E:\\Python_Titanic')

titanic_train = pd.read_csv('train.csv')
titanic_train.shape
titanic_train.info()

titanic_test = pd.read_csv('test.csv')
titanic_test.shape
titanic_test.info()

titanic_train1 = pd.get_dummies(titanic_train, columns = ['Pclass','Sex','Embarked'])
titanic_train1.info()

x_train = titanic_train1[['Pclass_1','Pclass_2','Pclass_3','SibSp','Parch','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S']]
y_train = titanic_train[['Survived']]

dt = tree.DecisionTreeClassifier()
param_grid = {'max_depth':[8,10,17],'min_samples_split':[2,3,5],'criterion':['gini','entropy']}
dt_grid = model_selection.GridSearchCV(dt,param_grid,cv=6,n_jobs=2)
dt_grid.fit(x_train,y_train)
dt_grid.param_grid
dt_grid.cv_results_
dt_grid.best_params_
dt_grid.best_score_
final_model = dt_grid.best_estimator_

dt_grid.score(x_train,y_train)

#objStringIO = io.StringIO()
#tree.export_graphviz(dt,out_file=objStringIO,feature_names=x_train.columns)
#file1 = pydot.graph_from_dot_data(objStringIO.getvalue())
#file1[0].write_pdf('Pagal_DT.pdf')

objStringIO = io.StringIO()
tree.export_graphviz(final_model ,out_file=objStringIO,feature_names=x_train.columns)
file1 = pydot.graph_from_dot_data(objStringIO.getvalue())
file1[0].write_pdf('Best_DT.pdf')

x_test = titanic_test[['Pclass','SibSp','Parch']]
titanic_test['Survived'] = dt.predict(x_test)
titanic_test.to_csv('Pagal.csv',columns=['PassengerId','Survived'],index=False)