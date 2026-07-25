import optuna
import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr


class OptunaTuner:
    def __init__(self, n_trials: int = 50):
        self.n_trials = n_trials
        self.best_params = None

    def objective(self, trial, X_train, y_train, X_val, y_val):
        params = {
            "objective": "lambdarank",
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
            "verbosity": -1,
        }

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        pred = model.predict(X_val)
        ic, _ = spearmanr(pred, y_val)
        return abs(ic)

    def tune(self, X_train, y_train, X_val, y_val):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials,
        )
        self.best_params = study.best_params
        return self.best_params
