import numpy as np
import pandas as pd
from model import Models
from ensemble import Ensemble
from param_opt import Optimizer
import random
import os
from feature import Feature
from controller import Controller

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

ct = Controller(1)

#ct.opt()
#ct.KFold_learn()
ct.stacking_learn(categorical_features, X_train, y_train)
#ct.stacking_predict(categorical_features, X_test)
#docker update --cpuset-cpus 0-6 kaggle
#docker container inspect --format='{{.HostConfig.CpusetCpus}}' kaggle