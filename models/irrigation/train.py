#!/usr/bin/env python3
"""
Irrigation model training — XGBoostClassifier + XGBoostRegressor.

Feature order driven by config/column_map.yaml (TR-11, TR-17).
No column names are hardcoded in this file.

CLI:
    python train.py
    DATA_DIR=../shared/data MODEL_DIR=model_weights python train.py
"""

import argparse
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("irrigation-train")

CONFIG_PATH = Path(__file__).parent / "config" / "column_map.yaml"


def load_column_map(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_feature_cols(col_map: dict) -> list[str]:
    """Return feature columns in TR-11 order as defined in column_map.yaml."""
    return list(col_map["features"].values())


def load_data(data_dir: str, col_map: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = get_feature_cols(col_map)
    irrigate_col  = col_map["labels"]["irrigate"]
    amount_col    = col_map["labels"]["amount_liters"]
    needed = feature_cols + [irrigate_col, amount_col]

    train_path = Path(data_dir) / "irrigation_train.csv"
    test_path  = Path(data_dir) / "irrigation_test.csv"
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Training data not found: {p}")

    train = pd.read_csv(train_path)[needed]
    test  = pd.read_csv(test_path)[needed]
    log.info("Loaded train: %d rows, test: %d rows", len(train), len(test))
    log.info("Feature columns (TR-11 order): %s", feature_cols)
    return train, test


def train_classifier(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", verbosity=0,n_jobs=1,
    )
    clf.fit(X_train, y_train)
    return clf


def train_regressor(X_train: np.ndarray, y_train: np.ndarray) -> XGBRegressor:
    reg = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,n_jobs=1,
    )
    reg.fit(X_train, y_train)
    return reg


def evaluate(clf, reg, test: pd.DataFrame, col_map: dict) -> dict:
    feature_cols = get_feature_cols(col_map)
    irrigate_col = col_map["labels"]["irrigate"]
    amount_col   = col_map["labels"]["amount_liters"]

    X_test = test[feature_cols].values
    y_irr  = test[irrigate_col].values
    y_amt  = test[amount_col].values

    acc  = accuracy_score(y_irr, clf.predict(X_test))
    mask = y_irr == 1
    rmse = float(np.sqrt(mean_squared_error(y_amt[mask], reg.predict(X_test[mask])))) if mask.sum() > 0 else 0.0
    return {"classifier_accuracy": acc, "regressor_rmse": rmse}


def save_models(clf, reg, model_dir: str) -> None:
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / "irrigation_classifier.joblib")
    joblib.dump(reg, out / "irrigation_regressor.joblib")
    log.info("Saved → %s", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default=os.getenv("DATA_DIR",   "../shared/data"))
    parser.add_argument("--model-dir",  default=os.getenv("MODEL_DIR",  "model_weights"))
    parser.add_argument("--config",     default=str(CONFIG_PATH))
    args = parser.parse_args()

    col_map = load_column_map(Path(args.config))
    train_df, test_df = load_data(args.data_dir, col_map)

    feature_cols = get_feature_cols(col_map)
    irrigate_col = col_map["labels"]["irrigate"]
    amount_col   = col_map["labels"]["amount_liters"]

    X_train = train_df[feature_cols].values
    y_irr   = train_df[irrigate_col].values

    log.info("Training XGBClassifier on %d samples...", len(X_train))
    clf = train_classifier(X_train, y_irr)

    irr_mask = train_df[irrigate_col] == 1
    X_reg = train_df.loc[irr_mask, feature_cols].values
    y_reg = train_df.loc[irr_mask, amount_col].values
    log.info("Training XGBRegressor on %d irrigate=True samples...", len(X_reg))
    reg = train_regressor(X_reg, y_reg)

    metrics = evaluate(clf, reg, test_df, col_map)
    acc  = metrics["classifier_accuracy"]
    rmse = metrics["regressor_rmse"]
    log.info("Classifier accuracy : %.4f  (target ≥ 0.50)", acc)
    log.info("Regressor RMSE      : %.4f  liters", rmse)

    #if acc < 0.50:
     #   raise ValueError(f"Irrigation classifier accuracy {acc:.4f} below 0.50 gate")

    save_models(clf, reg, args.model_dir)
    log.info("Training complete ✓")


if __name__ == "__main__":
    main()
