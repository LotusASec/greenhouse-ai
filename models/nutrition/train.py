#!/usr/bin/env python3
"""
Nutrition model training — RandomForestClassifier (4 classes).

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("nutrition-train")

CONFIG_PATH = Path(__file__).parent / "config" / "column_map.yaml"


def load_column_map(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_feature_cols(col_map: dict) -> list[str]:
    """Return feature columns in TR-11 order as defined in column_map.yaml."""
    return list(col_map["features"].values())


def load_data(data_dir: str, col_map: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = get_feature_cols(col_map)
    label_col    = col_map["labels"]["deficiency_class"]
    needed = feature_cols + [label_col]

    train_path = Path(data_dir) / "nutrition_train.csv"
    test_path  = Path(data_dir) / "nutrition_test.csv"
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Training data not found: {p}")

    train = pd.read_csv(train_path)[needed]
    test  = pd.read_csv(test_path)[needed]
    log.info("Loaded train: %d rows, test: %d rows", len(train), len(test))
    log.info("Feature columns (TR-11 order): %s", feature_cols)
    log.info("Label column: %s", label_col)
    log.info("Train class distribution:\n%s",
             train[label_col].value_counts().to_string())
    return train, test


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=42, n_jobs=-1, class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, test: pd.DataFrame, col_map: dict) -> dict:
    feature_cols = get_feature_cols(col_map)
    label_col    = col_map["labels"]["deficiency_class"]

    X_test = test[feature_cols].values
    y_test = test[label_col].values
    y_pred = clf.predict(X_test)

    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, zero_division=0)
    log.info("Classification report:\n%s", report)
    return {"weighted_f1": weighted_f1}


def save_model(clf, model_dir: str, class_mapping: dict) -> None:
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Artifact carries its own label semantics (TR-17) — inference must not
    # hardcode the int → class-name mapping.
    artifact = {"model": clf, "class_mapping": class_mapping}
    joblib.dump(artifact, out / "nutrition_model.joblib")
    log.info("Saved → %s (classes: %s)", out / "nutrition_model.joblib", class_mapping)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  default=os.getenv("DATA_DIR",  "../shared/data"))
    parser.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "model_weights"))
    parser.add_argument("--config",    default=str(CONFIG_PATH))
    args = parser.parse_args()

    col_map = load_column_map(Path(args.config))
    train_df, test_df = load_data(args.data_dir, col_map)

    feature_cols = get_feature_cols(col_map)
    label_col    = col_map["labels"]["deficiency_class"]

    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values

    log.info("Training RandomForestClassifier on %d samples...", len(X_train))
    clf = train_model(X_train, y_train)

    metrics = evaluate(clf, test_df, col_map)
    wf1 = metrics["weighted_f1"]
    log.info("Weighted F1: %.4f  (target ≥ 0.50)", wf1)

    if wf1 < 0.50:
        raise ValueError(f"Nutrition weighted F1 {wf1:.4f} below 0.50 gate")

    class_mapping = {int(k): str(v) for k, v in col_map["class_mapping"].items()}
    save_model(clf, args.model_dir, class_mapping)
    log.info("Training complete ✓")


if __name__ == "__main__":
    main()
