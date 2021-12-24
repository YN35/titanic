from sklearn import ensemble
from model import Models
import numpy as np
import pandas as pd
from util import Util

md = Models()
ut = Util()

class Ensemble():

    def __init__(self) -> None:
        pass

    def stacking(self, categorical_features, learn_type, fileID=0, X_train=None, y_train=None, X_test=None, fst_lay=['random_forest', 'light_gbm', 'xgboost', 'catboost'], snd_lay='light_gbm', enable_2ndorigx=True):
        #enable_2ndorigx:二層目にオリジナルの入力データを入力するか

        stack_oof_pred = []
        stack_pred = []
        for index, model_name in enumerate(fst_lay):
            
            if learn_type=='learn':
                cv_score, oof_pre, y_sub = md.KFold(categorical_features, model_name, learn_type, fileID=fileID+'0', X_train=X_train, y_train=y_train)
                stack_oof_pred = oof_pre if index == 0 else np.c_[stack_oof_pred, oof_pre]
            elif learn_type=='predict':
                cv_score, oof_pre, y_sub = md.KFold(categorical_features, model_name, learn_type, fileID=fileID+'0', X_test=X_test)
                stack_pred = y_sub if index == 0 else np.c_[stack_pred, y_sub]
            else:
                raise NameError("指定されたlearn_typeは存在しません")

        if enable_2ndorigx:
            X_train2 =  pd.concat([pd.DataFrame(stack_oof_pred), X_train], axis=1)
            X_test2 =  pd.concat([pd.DataFrame(stack_pred), X_test], axis=1)
        else:
            X_train2 = pd.DataFrame(stack_oof_pred)
            X_test2 = pd.DataFrame(stack_pred)
            categorical_features = []

        #二層目
        if learn_type=='learn':
            cv_score, oof_pre, y_sub = md.KFold(categorical_features, snd_lay, learn_type, fileID=fileID+'1', X_train=X_train2, y_train=y_train)
        elif learn_type=='predict':
            cv_score, oof_pre, y_sub = md.KFold(categorical_features, snd_lay, learn_type, fileID=fileID+'1', X_test=X_test2)
            y_sub = ut.data_conv(y_sub)

        return cv_score, oof_pre, y_sub

    def mean(self, categorical_features, learn_type, fileID=0, X_train=None, y_train=None, X_test=None, models=['random_forest', 'light_gbm', 'xgboost', 'catboost'], type='mean'):
        '''
        type:平均の取り方 
        mean -> 算術平均
        hmean -> 調和平均
        gmean -> 幾何平均
        '''

        stack_oof_pred = []
        stack_pred = []
        for index, model_name in enumerate(models):
            if learn_type=='learn':
                cv_score, oof_pre, y_sub = md.KFold(categorical_features, model_name, learn_type, fileID=fileID+'0', X_train=X_train, y_train=y_train)
                stack_oof_pred = oof_pre if index == 0 else np.c_[stack_oof_pred, oof_pre]
            elif learn_type=='predict':
                cv_score, oof_pre, y_sub = md.KFold(categorical_features, model_name, learn_type, fileID=fileID+'0', X_test=X_test)
                stack_pred = y_sub if index == 0 else np.c_[stack_pred, y_sub]
            else:
                raise NameError("指定されたlearn_typeは存在しません")

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

        cv_score = ut.accuracy_score(y_train, y_off)
        
        return cv_score, oof_pre, y_sub