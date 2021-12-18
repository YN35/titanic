import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from catboost import Pool
from catboost import CatBoostClassifier
import numpy as np
from util import Util

ut = Util()

class Models:
    def __init__(self) -> None:
        pass

    def KFold(self, X_train, y_train, X_test, categorical_features, model_name, n_splits=5):

        scores = []
        oof_pre = np.array([])
        valid_indexs = np.array([])
        y_preds = []
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
            X_tr = X_train.loc[train_index, :]
            X_val = X_train.loc[valid_index, :]
            y_tr = y_train[train_index]
            y_val = y_train[valid_index]

            if model_name == "random_forest":
                score, y_val_pre, y_pred = self.random_forest(X_tr, y_tr, X_valid=X_val, y_valid=y_val, X_test=X_test)
            elif model_name == "light_gbm":
                score, y_val_pre, y_pred = self.light_gbm(X_tr, y_tr, categorical_features, X_valid=X_val, y_valid=y_val, X_test=X_test)
            elif model_name == 'xgboost':
                score, y_val_pre, y_pred = self.xgboost(X_tr, y_tr, X_valid=X_val, y_valid=y_val, X_test=X_test)
            elif model_name == "catboost":
                score, y_val_pre, y_pred = self.catboost(X_tr, y_tr, categorical_features, X_valid=X_val, y_valid=y_val, X_test=X_test)
            elif model_name == "logistic_regression":
                score, y_val_pre, y_pred = self.logistic_regression(X_tr, y_tr, X_val, y_val, X_test=X_test)
            else:
                raise NameError("指定されたアルゴリズムは存在しません")

            scores.append(score)
            oof_pre = np.append(oof_pre,y_val_pre)
            valid_indexs = np.append(valid_indexs, valid_index)
            y_preds.append(y_pred)
            
        y_sub = sum(y_preds) / len(y_preds)
        y_sub = ut.data_conv(y_sub)

        oof_pre = oof_pre[np.argsort(valid_indexs)]

        cv_score = sum(scores) / len(scores)
        return cv_score, oof_pre, y_sub

    def random_forest(self, X_train, y_train, X_valid=None, y_valid=None, X_test=None, n_estimators=67, max_depth=6, random_state=0):
        """
        pandasでの教師データ
        パラメータ
        return valスコア(float)、その取り出し方での予測値
        """
        RandomForest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        RandomForest.fit(X_train, y_train)
        
        y_val_pre = RandomForest.predict(X_valid)
        y_val_pre = ut.data_conv(y_val_pre)
        score = accuracy_score(y_valid, y_val_pre)

        y_pred = RandomForest.predict(X_test) if not X_test is None else None
        y_pred = ut.data_conv(y_pred)

        return score, y_val_pre, y_pred

    def light_gbm(self, X_train, y_train, categorical_features, X_valid=None, y_valid=None, X_test=None, params = {'objective': 'binary','max_bin': 284,'learning_rate': 0.068,'num_leaves': 45}):
        """
        pandasでの教師データ
        categorical_features:カテゴリかる属性のカラム名を示したリスト
        パラメータ
        return valスコア(float), y_val_pre(valでの予測値), その取り出し方での予測値
        """

        lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)
        lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features)

        li_gbm = lgb.train(params, lgb_train,valid_sets=[lgb_train, lgb_eval],verbose_eval=10,num_boost_round=1000,early_stopping_rounds=10)

        y_val_pre = li_gbm.predict(X_valid, num_iteration=li_gbm.best_iteration)
        y_val_pre = ut.data_conv(y_val_pre)
        score = accuracy_score(y_valid, y_val_pre)

        y_pred = li_gbm.predict(X_test, num_iteration=li_gbm.best_iteration) if not X_test is None else None
        y_pred = ut.data_conv(y_pred)

        return score, y_val_pre, y_pred

    def xgboost(self, X_train, y_train, X_valid=None, y_valid=None, X_test=None, params = {'objective': 'reg:squarederror','silent':1, 'random_state':0,'learning_rate': 0.15, 'eval_metric': 'rmse',}, num_round = 450):
        """
        pandasでの教師データ
        categorical_features:カテゴリかる属性のカラム名を示したリスト
        パラメータ
        return valスコア(float), y_val_pre(valでの予測値), その取り出し方での予測値
        """

        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_valid, label=y_valid)
        test = xgb.DMatrix(X_test) if not X_test is None else None

        model = xgb.train(params,
                    train,#訓練データ
                    num_round,#設定した学習回数
                    early_stopping_rounds=20,
                    evals=[(train, 'train'), (valid, 'eval')],
                    )

        y_val_pre = model.predict(valid)
        y_val_pre = ut.data_conv(y_val_pre)
        score = accuracy_score(y_valid, y_val_pre)

        y_pred = model.predict(test) if not X_test is None else None
        y_pred = ut.data_conv(y_pred)

        return score, y_val_pre, y_pred

    def catboost(self, X_train, y_train, categorical_features, X_valid=None, y_valid=None, X_test=None, params ={'depth' : 3,'learning_rate' : 0.054,'early_stopping_rounds' : 9,'iterations' : 474, 'custom_loss' :['Accuracy'], 'random_seed' :0}):
        """
        pandasでの教師データ
        categorical_features:カテゴリかる属性のカラム名を示したリスト
        パラメータ
        return valスコア(float), y_val_pre(valでの予測値), その取り出し方での予測値
        """

        train = Pool(X_train, y_train, cat_features=categorical_features)
        eval = Pool(X_valid, y_valid, cat_features=categorical_features)

        model = CatBoostClassifier(custom_loss=['Accuracy'],random_seed=0)
        model = CatBoostClassifier(**params)
        cab = model.fit(train, eval_set=eval)
        
        y_val_pre = cab.predict(X_valid)
        y_val_pre = ut.data_conv(y_val_pre)
        score = accuracy_score(y_valid, y_val_pre)

        y_pred = cab.predict(X_test) if not X_test is None else None
        y_pred = ut.data_conv(y_pred)

        return score, y_val_pre, y_pred

    def logistic_regression(self, X_train, y_train, X_valid=None, y_valid=None, X_test=None):
        """
        pandasでの教師データ
        パラメータ
        return valスコア(float)、その取り出し方での予測値
        """
        model = LogisticRegression(penalty='l2', solver='sag', random_state=0)
        model.fit(X_train, y_train)

        y_val_pre = model.predict(X_valid)
        y_val_pre = ut.data_conv(y_val_pre)
        score = accuracy_score(y_valid, y_val_pre)

        y_pred = model.predict(X_test) if not X_test is None else None
        y_pred = ut.data_conv(y_pred)

        return score, y_val_pre, y_pred