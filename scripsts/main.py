import lightgbm
import numpy as np
import pandas as pd
from model import Models
from param_opt import Optimizer
import random
import os

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything

train = pd.read_csv('input/titanic/train.csv')
test = pd.read_csv('input/titanic/test.csv')
gender_submission = pd.read_csv('input/titanic/gender_submission.csv')

data = pd.concat([train, test], sort=False)

data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
data['Age'].fillna(data['Age'].median(), inplace=True)
data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
data['IsAlone'] = 0
data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1


delete_columns = ['Name', 'PassengerId', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)

train = data[:len(train)]
test = data[len(train):]

y_train = train['Survived']
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)

categorical_features = ['Embarked', 'Pclass', 'Sex']

md = Models()
op = Optimizer()

#print(op.param_opt('light_gbm', X_train, y_train))
#print(op.param_opt('random_forest', X_train, y_train))
#print(op.param_opt('catboost', X_train, y_train))

#K-foldでCVスコアを計算
print("CV :"+str(md.KFold(X_train, y_train, X_test, categorical_features, "random_forest"))+"-------------------------------------------------")

#random_forest 0.822635113928818
#light_gbm 0.8293829640323895
#catboost 0.8204004770573097