from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from sklearn.model_selection import KFold


class Models:
    def __init__(self) -> None:
        pass

    def KFold(self, X_train, y_train, X_test):


        return predict

    def random_forest(self, X_train, y_train, X_valid, y_valid, X_test=None, n_estimators=100, max_depth=2, random_state=0):
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

    def light_gbm(self, X_train, y_train, X_valid, y_valid, categorical_features, X_test=None, params = {'objective': 'binary','max_bin': 300,'learning_rate': 0.05,'num_leaves': 40}):
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