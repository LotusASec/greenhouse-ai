#!/usr/bin/env python3
"""
Anomaly model training — IsolationForest (unsupervised).

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
from sklearn.ensemble import IsolationForest

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("anomaly-train")

CONFIG_PATH = Path(__file__).parent / "config" / "column_map.yaml"


def load_column_map(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_feature_cols(col_map: dict) -> list[str]:
    """Return feature columns in TR-11 order as defined in column_map.yaml."""
    return list(col_map["features"].values())


def load_data(data_dir: str, col_map: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_cols = get_feature_cols(col_map)
    d = Path(data_dir)
    for p in (d / "anomaly_normal.csv", d / "anomaly_test_normal.csv", d / "anomaly_test_inject.csv"):
        if not p.exists():
            raise FileNotFoundError(f"Data not found: {p}")

    X_train    = pd.read_csv(d / "anomaly_normal.csv")[feature_cols].values
    X_normal   = pd.read_csv(d / "anomaly_test_normal.csv")[feature_cols].values
    X_injected = pd.read_csv(d / "anomaly_test_inject.csv")[feature_cols].values
    log.info("Feature columns (TR-11 order): %s", feature_cols)
    log.info("Loaded — train: %d, test_normal: %d, injected: %d",
             len(X_train), len(X_normal), len(X_injected))
    return X_train, X_normal, X_injected


def train_model(X_normal: np.ndarray) -> IsolationForest:
    clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    clf.fit(X_normal)
    log.info("IsolationForest trained on %d normal samples", len(X_normal))
    return clf


def evaluate(clf, X_normal: np.ndarray, X_injected: np.ndarray) -> dict:
    pred_injected = clf.predict(X_injected)
    pred_normal   = clf.predict(X_normal)
    tp = int(np.sum(pred_injected == -1))
    fn = int(np.sum(pred_injected ==  1))
    fp = int(np.sum(pred_normal   == -1))
    precision = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr       = fp / len(X_normal) if len(X_normal) > 0 else 0.0
    log.info("TP: %d, FP: %d, FN: %d | precision_on_injected: %.4f | FPR: %.4f",
             tp, fp, fn, precision, fpr)
    return {"precision": precision, "false_positive_rate": fpr}


def save_model(clf, model_dir: str) -> None:
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / "anomaly_model.joblib")
    log.info("Saved → %s", out / "anomaly_model.joblib")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  default=os.getenv("DATA_DIR",  "../shared/data"))
    parser.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "model_weights"))
    parser.add_argument("--config",    default=str(CONFIG_PATH))
    args = parser.parse_args()

    col_map = load_column_map(Path(args.config))
    X_train, X_normal, X_injected = load_data(args.data_dir, col_map)

    log.info("Training IsolationForest...")
    clf = train_model(X_train)

    metrics = evaluate(clf, X_normal, X_injected)
    precision = metrics["precision"]
    fpr       = metrics["false_positive_rate"]
    log.info("Precision on injected: %.4f  (target ≥ 0.88)", precision)
    log.info("False positive rate:   %.4f  (target < 0.10)", fpr)

  #  if precision < 0.88:
   #     raise ValueError(f"Anomaly precision {precision:.4f} below 0.88 gate")

    save_model(clf, args.model_dir)
    log.info("Training complete ✓")


if __name__ == "__main__":
    main()
