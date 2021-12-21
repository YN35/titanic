import numpy as np
import pandas as pd
from model import Models
from ensemble import Ensemble
from param_opt import Optimizer
import random
import os
from feature import Feature

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything

train = pd.read_csv('input/titanic/train.csv')
test = pd.read_csv('input/titanic/test.csv')

fe = Feature()

data = pd.concat([train, test], sort=False)

data = fe.conv_data(data, save_fet=False, save_name='feature')
categorical_features = ['Embarked', 'Pclass', 'Sex']

train = data[:len(train)]
test = data[len(train):]

y_train = train['Survived']
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)

md = Models()
op = Optimizer()
ens = Ensemble()

def opt():
    print(op.param_opt('light_gbm', X_train, y_train))

def KFold_learn():
    cv_score, y_val_pre, y_sub = md.KFold(categorical_features, 'light_gbm', 'learn', fileID=0, X_train=X_train, y_train=y_train)

    print('CV score-----------------------------------',cv_score)
    #random_forest 0.822635113928818
    #light_gbm 0.8293829640323895
    #catboost 0.8204004770573097
    #logistic_regression 0.6846023476241291
    #xgboost 0.8192643274119641
    #dnn 0.8159060950348378s

def KFold_predict():
    cv_score, y_val_pre, y_sub = md.KFold(categorical_features, 'light_gbm', 'predict', fileID=0, X_test=X_test)

    sub = pd.read_csv('input/titanic/gender_submission.csv')
    sub['Survived'] = y_sub
    sub.to_csv('submission.csv', index=False)


def stacking_learn():
    cv_score, _, y_sub = ens.stacking(categorical_features, 'learn', X_train=X_train, y_train=y_train, fst_lay=['random_forest', 'light_gbm', 'xgboost', 'catboost'], snd_lay='light_gbm', enable_2ndorigx=False)

    print('CV score-----------------------------------',cv_score)
    #0.8293829640323895 2ndlgtm
    #0.8237712635741635 2ndrandomforest

def stacking_predict():
    _, _, y_sub = ens.stacking(categorical_features, 'predict', X_test=X_test, fst_lay=['random_forest', 'light_gbm', 'xgboost', 'catboost'], snd_lay='light_gbm', enable_2ndorigx=False)

    sub = pd.read_csv('input/titanic/gender_submission.csv')
    sub['Survived'] = y_sub
    sub.to_csv('submission.csv', index=False)
    #0.8293829640323895

def mean():
    cv_score, _, y_sub = ens.mean(X_train, y_train, X_test, categorical_features, models=['random_forest', 'light_gbm', 'xgboost', 'catboost'], type='mean')
    #mean:0.8237934904601572 hmean:0.819304152637486 gmean:0.819304152637486
    print('CV score-----------------------------------',cv_score)

    sub = pd.read_csv('input/titanic/gender_submission.csv')
    sub['Survived'] = y_sub
    sub.to_csv('submission.csv', index=False)

#opt()
#KFold_learn()
#stacking_learn()
stacking_predict()
#docker update --cpuset-cpus 0-6 kaggle
#docker container inspect --format='{{.HostConfig.CpusetCpus}}' kaggle