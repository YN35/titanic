from model import Models
from sklearn.model_selection import train_test_split
import optuna
from sklearn.metrics import log_loss

md = Models()

#random_forest : {'n_estimators': 15, 'max_depth': 6}
#light_gbm : {'max_bin': 284, 'learning_rate': 0.06759289191947715, 'num_leaves': 45}
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
            study.optimize(self.objective_random_forest(X_train, y_train, X_valid, y_valid), n_trials=40)
            return study.best_params

        elif model_name == "light_gbm":
            study.optimize(self.objective_light_gbm(X_train, y_train, X_valid, y_valid, categorical_features), n_trials=40)
            return study.best_params

    def objective_random_forest(self, X_train, y_train, X_valid, y_valid):
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 300)
            max_depth = trial.suggest_int('max_depth', 1, 7)
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
            _, y_val_pre, _ = md.light_gbm(X_train, y_train, X_valid, y_valid, categorical_features, params=params)
            
            score = log_loss(y_valid, y_val_pre)

            return score
        return objective