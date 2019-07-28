# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:32:44 2019

@author: Gayatri
"""

import pandas as pd
from sklearn import tree,model_selection,preprocessing,ensemble
import os, io

os.chdir('E:\\Python_Titanic')

titanic_train = pd.read_csv('train.csv')
titanic_train.shape

titanic_test = pd.read_csv('test.csv')
titanic_test.shape

titanic_test.Survived = None
titanic_test.shape
titanic_test.info()

titanic_combined = pd.concat([titanic_train,titanic_test])
titanic_combined.info()

#Extract and create title from name column
def extract_title(name):
    return name.split(',')[1].split('.')[0].strip()
    
#map function applies a passed in function to each item in an iterable object and returns a list containing all function call results.
titanic_combined['title'] = titanic_combined['Name'].map(extract_title)

mean_imputer = preprocessing.Imputer()
mean_imputer.fit(titanic_combined[['Age','Fare']])

#titanic_train['Title'] = titanic_train['Name'].map(extract_title)
#print(titanic_train['Title'].unique())
#titanic_test['Title'] = titanic_test['Name'].map(extract_title)
#print(titanic_test['Title'].unique())

#titanic_combined.Age[titanic_combined['Age'].isnull()] = titanic_combined['Age'].mean()
#titanic_combined.Fare[titanic_combined['Fare'].isnull()] = titanic_combined['Fare'].mean()

def convert_age(age):
    if (age>=0 and age <=10):
        return 'Child'
    elif (age<=25):
        return 'Young'
    elif (age<=50):
        return 'Middle'
    else:
        return 'Old'
    
#Convert 'age' numerical column to 'Age1' categorical column
titanic_combined['Age_cat'] = titanic_combined['Age'].map(convert_age)
titanic_combined['Age_cat'].head(10)

#Create a new column FamilySize by combining SibSp and Parch and see we get any additioanl pattern recognition than individual
titanic_combined['FamilySize'] = titanic_combined['SibSp'] + titanic_combined['Parch'] + 1

def convert_familysize(size):
    if (size == 1):
        return 'Single'
    elif (size >= 3):
        return 'Small'
    elif (size >= 6):
        return 'Medium'
    else:
        return 'Large'
    
titanic_combined['FamilySize_cate'] = titanic_combined['FamilySize'].map(convert_familysize)

titanic_combined_new = pd.get_dummies(titanic_combined,columns=['Age_cat','Embarked','FamilySize_cate','Pclass','title','Sex'])
titanic_combined_new.shape
titanic_combined_new.info()

titanic_combined_new1 = titanic_combined_new.drop(['PassengerId','Survived','Ticket','Name','Cabin','Age'],axis=1,inplace=False)

x_train = titanic_combined_new1[0:titanic_train.shape[0]]
y_train = titanic_train['Survived']

#oob scrore is computed as part of model construction process

dt = tree.DecisionTreeClassifier(random_state =1)
bt_estimator = ensemble.BaggingClassifier(base_estimator = dt, random_state=102)
param_grid = {'n_estimators':[5,6] ,'base_estimator__max_depth':list(range(3,6)),'base_estimator__min_samples_split':list(range(2,7))}
bt_grid_estimator = model_selection.GridSearchCV(bt_estimator,param_grid,cv=10,n_jobs = 4)
bt_grid_estimator.fit(x_train,y_train)
bt_grid_estimator.cv_results_
bt_grid_estimator.best_params_
bt_grid_estimator.best_score_
bt_grid_estimator.score(x_train,y_train)

#Explore feature importances calculated by decision tree algorithm
#best_estimator_ gives final best parameters.
#feature_importances_: Every feature has an importance with a priority number. Now I want to use best estimator along with very very importance features
#Let's create a DataFrame with features and their importances.

x_test = titanic_combined_new1[titanic_train.shape[0]:]
titanic_test['Survived'] = bt_grid_estimator.predict(x_test)
titanic_test.to_csv('BaggingClassifier_Submission.csv',columns=['PassengerId','Survived'],index=False)