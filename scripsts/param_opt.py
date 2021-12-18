from model import Models
from sklearn.model_selection import train_test_split
import optuna
from sklearn.metrics import log_loss

md = Models()

#random_forest : {'n_estimators': 67, 'max_depth': 6}
#light_gbm : {'max_bin': 284, 'learning_rate': 0.06759289191947715, 'num_leaves': 45}
#xgboost : {'learning_rate': 0.180343853211702, 'num_round': 394}
#catboost : {'depth': 3, 'learning_rate': 0.053925065258405916, 'early_stopping_rounds': 9, 'iterations': 474}
class Optimizer():
    def __init__(self) -> None:
        pass

    def param_opt(self, model_name, X_train, y_train, categorical_features=None):
        """
        パラメータオプティマイザー
        model name list: random_forest, light_gbm
        """
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3,random_state=0, stratify=y_train)
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        if model_name == "random_forest":
            study.optimize(self.objective_random_forest(X_train, y_train, X_valid, y_valid), n_trials=80)
            return study.best_params

        elif model_name == "light_gbm":
            study.optimize(self.objective_light_gbm(X_train, y_train, X_valid, y_valid, categorical_features), n_trials=80)
            return study.best_params

        elif model_name == "xgboost":
            study.optimize(self.objective_xgboost(X_train, y_train, X_valid, y_valid), n_trials=80)
            return study.best_params

        elif model_name == "catboost":
            study.optimize(self.objective_catboost(X_train, y_train, X_valid, y_valid, categorical_features), n_trials=80)
            return study.best_params
            
        elif model_name == 'logistic_regression':
            raise NameError('logistic_regressionはパラメータが存在しないのでサポートしていません')

    def objective_random_forest(self, X_train, y_train, X_valid, y_valid):
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 300)
            max_depth = trial.suggest_int('max_depth', 1, 15)
            _, y_val_pre, _ = md.random_forest(X_train, y_train, X_valid, y_valid, n_estimators=n_estimators, max_depth=max_depth, random_state=0)
            
            score = log_loss(y_valid, y_val_pre)

            return score
        return objective

    def objective_light_gbm(self, X_train, y_train, X_valid, y_valid, categorical_features):
        def objective(trial):
            params = {
            'objective': 'binary',
            'max_bin': trial.suggest_int('max_bin', 255, 500),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 32, 128),
            }
            _, y_val_pre, _ = md.light_gbm(X_train, y_train, categorical_features, X_valid=X_valid, y_valid=y_valid, params=params)
            
            score = log_loss(y_valid, y_val_pre)

            return score
        return objective

    def objective_xgboost(self, X_train, y_train, X_valid, y_valid):
        def objective(trial):
            params = {'objective': 'reg:squarederror',
                    'silent':1, 
                    'random_state':0,
                    'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.2), 
                    'eval_metric': 'rmse',
            }
            num_round = trial.suggest_int('num_round', 100, 900)
            _, y_val_pre, _ = md.xgboost(X_train, y_train, X_valid=X_valid, y_valid=y_valid, params=params, num_round=num_round)
            
            score = log_loss(y_valid, y_val_pre)

            return score
        return objective

    def objective_catboost(self, X_train, y_train, X_valid, y_valid, categorical_features):
        def objective(trial):
            params = {
                'depth' : trial.suggest_int('depth', 1, 15),                  # 木の深さ
                'learning_rate' : trial.suggest_uniform('learning_rate', 0.01, 0.1),       # 学習率
                'early_stopping_rounds' : trial.suggest_int('early_stopping_rounds', 3, 20),
                'iterations' : trial.suggest_int('iterations', 50, 500), 
                'custom_loss' :['Accuracy'], 
                'random_seed' :0
            }
            _, y_val_pre, _ = md.catboost(X_train, y_train, categorical_features, X_valid=X_valid, y_valid=y_valid, params=params)
            
            score = log_loss(y_valid, y_val_pre)

            return score
        return objective

    #LogisticRegressionはパラメータがない