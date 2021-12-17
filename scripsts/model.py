from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from catboost import Pool
from catboost import CatBoostClassifier
import numpy as np

class Models:
    def __init__(self) -> None:
        pass

    def KFold(self, X_train, y_train, X_test, categorical_features, model_name, n_splits=5):

        scores = []
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
            X_tr = X_train.loc[train_index, :]
            X_val = X_train.loc[valid_index, :]
            y_tr = y_train[train_index]
            y_val = y_train[valid_index]

            if model_name == "random_forest":
                score, _, _ = self.random_forest(X_tr, y_tr, X_val, y_val)
            elif model_name == "light_gbm":
                score, _, _ = self.light_gbm(X_tr, y_tr, X_val, y_val, categorical_features)
            elif model_name == "catboost":
                score, _, _ = self.catboost(X_tr, y_tr, X_val, y_val, categorical_features)
            else:
                raise NameError("指定されたアルゴリズムは存在しません")

            scores.append(score)

        cv_score = sum(scores) / len(scores)
        return cv_score

    def random_forest(self, X_train, y_train, X_valid, y_valid, X_test=None, n_estimators=67, max_depth=6, random_state=0):
        """
        pandasでの教師データ
        パラメータ
        return valスコア(float)、その取り出し方での予測値
        """
        RandomForest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        RandomForest.fit(X_train, y_train)

        y_val_pre = RandomForest.predict(X_valid)
        y_val_pre = (y_val_pre > 0.5).astype(int)
        score = accuracy_score(y_valid, y_val_pre)

        y_pred = RandomForest.predict(X_test) if not X_test==None else None

        return score, y_val_pre, y_pred

    def light_gbm(self, X_train, y_train, X_valid, y_valid, categorical_features, X_test=None, params = {'objective': 'binary','max_bin': 284,'learning_rate': 0.068,'num_leaves': 45}):
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
        y_val_pre = (y_val_pre > 0.5).astype(int)
        score = accuracy_score(y_valid, y_val_pre)

        y_pred = li_gbm.predict(X_test, num_iteration=li_gbm.best_iteration) if not X_test==None else None

        return score, y_val_pre, y_pred

    def catboost(self, X_train, y_train, X_valid, y_valid, categorical_features, X_test=None, params ={'depth' : 3,'learning_rate' : 0.054,'early_stopping_rounds' : 9,'iterations' : 474, 'custom_loss' :['Accuracy'], 'random_seed' :0}):
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
        y_val_pre = (y_val_pre > 0.5).astype(int)
        score = accuracy_score(y_valid, y_val_pre)

        y_pred = cab.predict(X_test) if not X_test==None else None

        return score, y_val_pre, y_pred