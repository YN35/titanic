from sklearn import ensemble
from model import Models
import numpy as np
import pandas as pd

md = Models()

class Ensemble():

    def __init__(self) -> None:
        pass

    def stacking(self, X_train, y_train, X_test, categorical_features, fst_lay=['random_forest', 'light_gbm', 'xgboost', 'catboost'], snd_lay='light_gbm'):

        for index, model_name in enumerate(fst_lay):

            cv_score, oof_pre, y_sub = md.KFold(X_train, y_train, X_test, categorical_features, model_name)

            if index == 0:
                stack_oof_pred = oof_pre
                stack_pred = y_sub
            else:
                stack_oof_pred = np.c_[stack_oof_pred, oof_pre]
                stack_pred = np.c_[stack_pred, y_sub]

        categorical_features = []
        cv_score, oof_pre, y_sub = md.KFold(pd.DataFrame(stack_oof_pred), y_train, pd.DataFrame(stack_pred), categorical_features, snd_lay)
        
        y_sub = sum(y_sub) / len(y_sub)
        y_sub = (y_sub > 0.5).astype(int)

        return cv_score, oof_pre, y_sub

