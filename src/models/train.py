# pyrefly: ignore [missing-import]
import os

import joblib
# pyrefly: ignore [missing-import]
import lightgbm as lgb
import xgboost as xgb

MODEL_DIR = "data/models/"


def train_xgboost_baseline(X_train, Y_train):
    print("Training xgboost model:")

    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )

    model.fit(X_train, Y_train, eval_set=[(X_train, Y_train)], verbose=True)

    print("Xgboost baseline trained")

    return model


def train_lightgbm_baseline(X_train, Y_train):
    print("Training LightGBM baseline...")

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosre=1,
    )

    model.fit(X_train, Y_train)

    return model


def save_model(model, name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"Model saved to {path}")
    return path


def load_model(name):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    return joblib.load(path)


import numpy as np
# pyrefly: ignore [missing-import]
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_xgboost(X_train, Y_train, n_trials=50):

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "eval_metric": "aucpr",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

        model = xgb.XGBClassifier(**params)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(
            model, X_train, Y_train, cv=cv, scoring="average_precision", n_jobs=-1
        )

        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest AUC-PR (CV): {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f" {k}: {v}")

    best_params = study.best_params
    best_params.update(
        {
            "eval_metric": "aucpr",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
    )

    print("\nTraining final model with best params...")
    tuned_model = xgb.XGBClassifier(**best_params)
    tuned_model.fit(X_train, Y_train)

    return tuned_model, study
