"""
Irrigation model inference — XGBoost classifier + regressor.

Feature order driven by config/column_map.yaml (TR-11, TR-17).
No column names are hardcoded in this file.
Output schema: MASTER_SPEC §3.3 (frozen)
"""

import os
import time
from pathlib import Path

import joblib
import numpy as np
import yaml

CONFIG_PATH = Path(__file__).parent / "config" / "column_map.yaml"


def _load_column_map(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class IrrigationInference:
    def __init__(self, model_dir: str, config_path: Path = CONFIG_PATH) -> None:
        col_map = _load_column_map(config_path)
        # Feature order = key order in column_map.yaml (TR-11)
        self._feature_keys = list(col_map["features"].keys())

        clf_path = Path(model_dir) / "irrigation_classifier.joblib"
        reg_path = Path(model_dir) / "irrigation_regressor.joblib"
        for p in (clf_path, reg_path):
            if not p.exists():
                raise FileNotFoundError(f"Model not found: {p}. Run train.py first.")

        self._clf = joblib.load(clf_path)
        self._reg = joblib.load(reg_path)
        self.model_name = "xgboost_irrigation"
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def predict(self, input_data: dict) -> dict:
        """Input/output per MASTER_SPEC §3.3."""
        t_start = time.perf_counter()

        sensors = input_data["sensors"]
        # Build feature vector using key order from column_map.yaml (TR-11)
        X = np.array([[sensors[k] for k in self._feature_keys]])

        irrigate_int  = int(self._clf.predict(X)[0])
        irrigate_bool = bool(irrigate_int)
        proba         = self._clf.predict_proba(X)[0]
        confidence    = float(proba[irrigate_int])

        amount_liters = float(np.clip(self._reg.predict(X)[0], 0.0, 10.0)) if irrigate_bool else 0.0

        return {
            "model":             self.model_name,
            "node_id":           input_data["node_id"],
            "timestamp":         input_data["timestamp"],
            "irrigate":          irrigate_bool,
            "amount_liters":     round(amount_liters, 4),
            "confidence":        round(confidence, 4),
            "inference_time_ms": round((time.perf_counter() - t_start) * 1000, 3),
        }
