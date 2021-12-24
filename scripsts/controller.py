import numpy as np
import pandas as pd
from model import Models
from ensemble import Ensemble
from param_opt import Optimizer


md = Models()
op = Optimizer()
ens = Ensemble()

class Controller():

    def __init__(self,ID) -> None:
        self.ID = str(ID)

    def opt(self, X_train, y_train):
        print(op.param_opt('light_gbm', X_train, y_train))

    def KFold_learn(self, categorical_features, X_train, y_train):
        cv_score, y_val_pre, y_sub = md.KFold(categorical_features, 'light_gbm', 'learn', fileID=self.ID+'0', X_train=X_train, y_train=y_train)

        print('CV score-----------------------------------',cv_score)
        #random_forest 0.822635113928818
        #light_gbm 0.8293829640323895
        #catboost 0.8204004770573097
        #logistic_regression 0.6846023476241291
        #xgboost 0.8192643274119641
        #dnn 0.8159060950348378s

    def KFold_predict(self, categorical_features, X_test):
        cv_score, y_val_pre, y_sub = md.KFold(categorical_features, 'light_gbm', 'predict', fileID=self.ID+'0', X_test=X_test)

        sub = pd.read_csv('input/titanic/gender_submission.csv')
        sub['Survived'] = y_sub
        sub.to_csv('submission.csv', index=False)

    def stacking_learn(self, categorical_features, X_train, y_train, fst_lay=['random_forest', 'light_gbm', 'xgboost', 'catboost'], snd_lay='light_gbm', enable_2ndorigx=False):
        cv_score, _, y_sub = ens.stacking(categorical_features, 'learn', fileID=self.ID, X_train=X_train, y_train=y_train, fst_lay=fst_lay, snd_lay=snd_lay, enable_2ndorigx=enable_2ndorigx)

        print('CV score-----------------------------------',cv_score)
        #0.8293829640323895 2ndlgtm
        #0.8237712635741635 2ndrandomforest

    def stacking_predict(self, categorical_features, X_test, fst_lay=['random_forest', 'light_gbm', 'xgboost', 'catboost'], snd_lay='light_gbm', enable_2ndorigx=False):
        _, _, y_sub = ens.stacking(categorical_features, 'predict', fileID=self.ID, X_test=X_test, fst_lay=fst_lay, snd_lay=snd_lay, enable_2ndorigx=enable_2ndorigx)

        sub = pd.read_csv('input/titanic/gender_submission.csv')
        sub['Survived'] = y_sub
        sub.to_csv('submission.csv', index=False)
        #0.8293829640323895

    def mean_learn(self, categorical_features, learn_type, X_train, y_train, models=['random_forest', 'light_gbm', 'xgboost', 'catboost'], type='mean'):
        cv_score, _, y_sub = ens.mean(categorical_features, learn_type, fileID=self.ID, X_train=X_train, y_train=y_train, models=models, type=type)
        #mean:0.8237934904601572 hmean:0.819304152637486 gmean:0.819304152637486
        print('CV score-----------------------------------',cv_score)

    def mean_predict(self, categorical_features, learn_type, X_test, models=['random_forest', 'light_gbm', 'xgboost', 'catboost'], type='mean'):
        cv_score, _, y_sub = ens.mean(categorical_features, learn_type, fileID=self.ID, X_test=X_test, models=models, type=type)

        sub = pd.read_csv('input/titanic/gender_submission.csv')
        sub['Survived'] = y_sub
        sub.to_csv('submission.csv', index=False)