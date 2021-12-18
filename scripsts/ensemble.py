from sklearn import ensemble
from model import Models
import numpy as np
import pandas as pd
from util import Util
from sklearn.metrics import accuracy_score

md = Models()
ut = Util()

class Ensemble():

    def __init__(self) -> None:
        pass

    def stacking(self, X_train, y_train, X_test, categorical_features, fst_lay=['random_forest', 'light_gbm', 'xgboost', 'catboost'], snd_lay='light_gbm', enable_2ndorigx=True):
        #enable_2ndorigx:二層目にオリジナルの入力データを入力するか

        for index, model_name in enumerate(fst_lay):

            cv_score, oof_pre, y_sub = md.KFold(X_train, y_train, X_test, categorical_features, model_name)

            if index == 0:
                stack_oof_pred = oof_pre
                stack_pred = y_sub
            else:
                stack_oof_pred = np.c_[stack_oof_pred, oof_pre]
                stack_pred = np.c_[stack_pred, y_sub]

        if enable_2ndorigx:
            X_train2 =  pd.concat([pd.DataFrame(stack_oof_pred), X_train], axis=1)
            X_test2 =  pd.concat([pd.DataFrame(stack_pred), X_test], axis=1)
        else:
            X_train2 = pd.DataFrame(stack_oof_pred)
            X_test2 = pd.DataFrame(stack_pred)
            categorical_features = []

        cv_score, oof_pre, y_sub = md.KFold(X_train2, y_train, X_test2, categorical_features, snd_lay)
        
        y_sub = ut.data_conv(y_sub)

        return cv_score, oof_pre, y_sub

    def mean(self, X_train, y_train, X_test, categorical_features, models=['random_forest', 'light_gbm', 'xgboost', 'catboost'], type='mean'):
        '''
        type:平均の取り方 
        mean -> 算術平均
        hmean -> 調和平均
        gmean -> 幾何平均
        '''

        for index, model_name in enumerate(models):

            cv_score, oof_pre, y_sub = md.KFold(X_train, y_train, X_test, categorical_features, model_name)
            if index == 0:
                stack_oof_pred = oof_pre
                stack_pred = y_sub
            else:
                stack_oof_pred = np.c_[stack_oof_pred, oof_pre]
                stack_pred = np.c_[stack_pred, y_sub]

        if type == 'mean':
            y_off = np.average(stack_oof_pred, axis=1)
            y_sub = np.average(stack_pred, axis=1)
        elif type == 'hmean':
            from scipy.stats import hmean
            y_off = hmean(stack_oof_pred, axis = 1)
            y_sub = hmean(stack_pred, axis = 1)
        elif type == 'gmean':
            from scipy.stats.mstats import gmean
            y_off = gmean(stack_oof_pred, axis = 1)
            y_sub = gmean(stack_pred, axis = 1)
        
        y_off = ut.data_conv(y_off)
        y_sub = ut.data_conv(y_sub)

        cv_score = accuracy_score(y_train, y_off)
        
        return cv_score, oof_pre, y_sub