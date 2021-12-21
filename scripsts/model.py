import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from catboost import Pool
from catboost import CatBoostClassifier
import numpy as np
from util import Util
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import pickle


ut = Util()

class Models:
    def __init__(self) -> None:
        pass

    def select_model(self, categorical_features, model_name, learn_type, fileID=0, X_train=None, y_train=None, X_valid=None, y_valid=None, X_test=None):
        if model_name == "random_forest":
            score, y_val_pre, y_pred = self.random_forest(learn_type, fileID=fileID, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test)
        elif model_name == "light_gbm":
            score, y_val_pre, y_pred = self.light_gbm(categorical_features, learn_type, fileID=fileID, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test)
        elif model_name == 'xgboost':
            score, y_val_pre, y_pred = self.xgboost(learn_type, fileID=fileID, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test)
        elif model_name == "catboost":
            score, y_val_pre, y_pred = self.catboost(categorical_features, learn_type, fileID=fileID, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test)
        elif model_name == "logistic_regression":
            score, y_val_pre, y_pred = self.logistic_regression(learn_type, fileID=fileID, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test)
        elif model_name == "dnn":
            score, y_val_pre, y_pred = self.dnn(learn_type, fileID=fileID, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test)
        else:
            raise NameError("指定されたアルゴリズムは存在しません")

        return score, y_val_pre, y_pred

    def KFold(self, categorical_features, model_name, learn_type, fileID=0, X_train=None, y_train=None, X_test=None, n_splits=5):

        cv_score, oof_pre, y_sub = None, None, None
        scores = []
        oof_pre = np.array([])
        valid_indexs = np.array([])
        y_preds = []
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        if learn_type=='learn':
            for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
                X_tr = X_train.loc[train_index, :]
                X_val = X_train.loc[valid_index, :]
                y_tr = y_train[train_index]
                y_val = y_train[valid_index]

                score, y_val_pre, _ = self.select_model(categorical_features, model_name, 'learn', fileID=str(fileID)+str(fold_id),X_train=X_tr, y_train=y_tr, X_valid=X_val, y_valid=y_val)

                scores.append(score)
                oof_pre = np.append(oof_pre,y_val_pre)
                valid_indexs = np.append(valid_indexs, valid_index)

            oof_pre = oof_pre[np.argsort(valid_indexs)]
            cv_score = sum(scores) / len(scores)
        elif learn_type=='predict':
            for fold_id in range(n_splits):
                score, y_val_pre, y_pred = self.select_model(categorical_features, model_name, 'predict', fileID=str(fileID)+str(fold_id), X_test=X_test)

                oof_pre = np.append(oof_pre,y_val_pre)
                y_preds.append(y_pred)
            
            y_sub = sum(y_preds) / len(y_preds)
            y_sub = ut.data_conv(y_sub)

        return cv_score, oof_pre, y_sub

    def random_forest(self, learn_type, fileID=0, X_train=None, y_train=None, X_valid=None, y_valid=None, X_test=None, n_estimators=67, max_depth=6, random_state=0):
        """
        pandasでの教師データ
        パラメータ
        return valスコア(float)、その取り出し方での予測値
        """
        score, y_val_pre, y_pred = None, None, None
        if learn_type=='predict':
            with open('RandomForest'+str(fileID)+'.pickle', 'rb') as web:
                RandomForest = pickle.load(web)
            y_pred = RandomForest.predict(X_test)
            y_pred = ut.data_conv(y_pred)
        elif learn_type=='learn':
            RandomForest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
            RandomForest.fit(X_train, y_train)
            y_val_pre = RandomForest.predict(X_valid)
            y_val_pre = ut.data_conv(y_val_pre)
            score = ut.accuracy_score(y_valid, y_val_pre)
            with open('RandomForest'+str(fileID)+'.pickle', 'wb') as web:
                pickle.dump(RandomForest , web)

        return score, y_val_pre, y_pred

    def light_gbm(self, categorical_features, learn_type, fileID=0, X_train=None, y_train=None, X_valid=None, y_valid=None, X_test=None, params = {'objective': 'binary','max_bin': 284,'learning_rate': 0.068,'num_leaves': 45}):
        """
        pandasでの教師データ
        categorical_features:カテゴリかる属性のカラム名を示したリスト
        パラメータ
        return valスコア(float), y_val_pre(valでの予測値), その取り出し方での予測値
        """
        score, y_val_pre, y_pred = None, None, None
        if learn_type=='predict':
            with open('light_gbm'+str(fileID)+'.pickle', 'rb') as web:
                model = pickle.load(web)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)
            y_pred = ut.data_conv(y_pred)
        elif learn_type=='learn':
            lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)
            lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features)
            model = lgb.train(params, lgb_train,valid_sets=[lgb_train, lgb_eval],verbose_eval=10,num_boost_round=1000,early_stopping_rounds=10)

            y_val_pre = model.predict(X_valid, num_iteration=model.best_iteration)
            y_val_pre = ut.data_conv(y_val_pre)
            score = ut.accuracy_score(y_valid, y_val_pre)
            with open('light_gbm'+str(fileID)+'.pickle', 'wb') as web:
                pickle.dump(model , web)

        return score, y_val_pre, y_pred

    def xgboost(self, learn_type, fileID=0, X_train=None, y_train=None, X_valid=None, y_valid=None, X_test=None, params = {'objective': 'reg:squarederror','silent':1, 'random_state':0,'learning_rate': 0.15, 'eval_metric': 'rmse',}, num_round = 450):
        """
        pandasでの教師データ
        categorical_features:カテゴリかる属性のカラム名を示したリスト
        パラメータ
        return valスコア(float), y_val_pre(valでの予測値), その取り出し方での予測値
        """
        score, y_val_pre, y_pred = None, None, None
        if learn_type=='predict':
            with open('xgboost'+str(fileID)+'.pickle', 'rb') as web:
                model = pickle.load(web)
            test = xgb.DMatrix(X_test)

            y_pred = model.predict(test)
            y_pred = ut.data_conv(y_pred)
        elif learn_type=='learn':
            train = xgb.DMatrix(X_train, label=y_train)
            valid = xgb.DMatrix(X_valid, label=y_valid)
            model = xgb.train(params,
                    train,#訓練データ
                    num_round,#設定した学習回数
                    early_stopping_rounds=20,
                    evals=[(train, 'train'), (valid, 'eval')],
                    )
            y_val_pre = model.predict(valid)
            y_val_pre = ut.data_conv(y_val_pre)
            score = ut.accuracy_score(y_valid, y_val_pre)
            with open('xgboost'+str(fileID)+'.pickle', 'wb') as web:
                pickle.dump(model, web)

        return score, y_val_pre, y_pred

    def catboost(self, categorical_features, learn_type, fileID=0, X_train=None, y_train=None, X_valid=None, y_valid=None, X_test=None, params ={'depth' : 3,'learning_rate' : 0.054,'early_stopping_rounds' : 9,'iterations' : 474, 'custom_loss' :['Accuracy'], 'random_seed' :0}):
        """
        pandasでの教師データ
        categorical_features:カテゴリかる属性のカラム名を示したリスト
        パラメータ
        return valスコア(float), y_val_pre(valでの予測値), その取り出し方での予測値
        """
        score, y_val_pre, y_pred = None, None, None
        if learn_type=='predict':
            with open('catboost'+str(fileID)+'.pickle', 'rb') as web:
                model = pickle.load(web)
            y_pred = model.predict(X_test)
            y_pred = ut.data_conv(y_pred)
        elif learn_type=='learn':
            train = Pool(X_train, y_train, cat_features=categorical_features)
            eval = Pool(X_valid, y_valid, cat_features=categorical_features)
            cab = CatBoostClassifier(custom_loss=['Accuracy'],random_seed=0)
            cab = CatBoostClassifier(**params)
            model = cab.fit(train, eval_set=eval)

            y_val_pre = model.predict(X_valid)
            y_val_pre = ut.data_conv(y_val_pre)
            score = ut.accuracy_score(y_valid, y_val_pre)
            with open('catboost'+str(fileID)+'.pickle', 'wb') as web:
                pickle.dump(model , web)

        return score, y_val_pre, y_pred

    def logistic_regression(self, learn_type, fileID=0, X_train=None, y_train=None, X_valid=None, y_valid=None, X_test=None):
        """
        pandasでの教師データ
        パラメータ
        return valスコア(float)、その取り出し方での予測値
        """
        score, y_val_pre, y_pred = None, None, None
        if learn_type=='predict':
            with open('logistic_regression'+str(fileID)+'.pickle', 'rb') as web:
                model = pickle.load(web)
            y_pred = model.predict(X_test)
            y_pred = ut.data_conv(y_pred)
        elif learn_type=='learn':
            model = LogisticRegression(penalty='l2', solver='sag', random_state=0)
            model.fit(X_train, y_train)
            y_val_pre = model.predict(X_valid)
            y_val_pre = ut.data_conv(y_val_pre)
            score = ut.accuracy_score(y_valid, y_val_pre)
            with open('logistic_regression'+str(fileID)+'.pickle', 'wb') as web:
                pickle.dump(model , web)

        return score, y_val_pre, y_pred

    def dnn(self, learn_type, fileID=0, X_train=None, y_train=None, X_valid=None, y_valid=None, X_test=None):

        score, y_val_pre, y_pred = None, None, None

        lr_schedule=tf.keras.optimizers.schedules.ExponentialDecay( \
                    initial_learning_rate=0.001, #初期の学習率
                    decay_steps=3000, #減衰ステップ数
                    decay_rate=0.01, #最終的な減衰率 
                    staircase=True)

        model=Sequential()
        model.add(Dense(len(X_train.columns),input_shape=(len(X_train.columns),),activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001), #重みの正則化考慮
                    kernel_initializer='random_uniform',
                    bias_initializer='zero'))
                    
        model.add(BatchNormalization()) #バッチ正規化
        model.add(Dropout(0.1)) # ドロップアウト層・ドロップアウトさせる割合
        model.add(Dense(int(len(pd.DataFrame(X_train).columns)/2),activation='sigmoid'))

        model.add(BatchNormalization()) #バッチ正規化
        model.add(Dropout(0.1)) # ドロップアウト層・ドロップアウトさせる割合
        model.add(Dense(int(len(pd.DataFrame(X_train).columns)/2),activation='sigmoid'))

        model.add(BatchNormalization()) #バッチ正規化
        model.add(Dropout(0.1)) # ドロップアウト層・ドロップアウトさせる割合
        model.add(Dense(len(pd.DataFrame(y_train).columns),activation='sigmoid'))
        Ecall=EarlyStopping(monitor='val_loss',patience=1000,restore_best_weights=False)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr_schedule))
        model.summary()

        if learn_type=='predict':
            model.load_weights('dnn'+str(fileID)+'.h5')
            y_pred = model.predict(X_test)
            y_pred = ut.data_conv(y_pred)
        elif learn_type=='learn':
            res=model.fit(X_train.values,y_train.values,epochs=10000,callbacks=[Ecall],verbose=1,validation_data=(X_valid.values,y_valid.values))
            y_val_pre = model.predict(X_valid)
            y_val_pre = ut.data_conv(y_val_pre)
            score = ut.accuracy_score(y_valid, y_val_pre)
            model.save_weights('dnn'+str(fileID)+'.h5')

        return score, y_val_pre, y_pred