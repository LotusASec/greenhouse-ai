"""
Anomaly model inference — IsolationForest (unsupervised).

Feature order driven by config/column_map.yaml (TR-11, TR-17).
No column names are hardcoded in this file.
Output schema: MASTER_SPEC §3.5 (frozen)
"""

import time
from pathlib import Path

import joblib
import numpy as np
import yaml

CONFIG_PATH = Path(__file__).parent / "config" / "column_map.yaml"


def _load_column_map(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _sigmoid(x: float) -> float:
    """1/(1+exp(x)) — decreasing: normal (raw>0) → low score, anomaly (raw<0) → high score."""
    return float(1.0 / (1.0 + np.exp(x)))


class AnomalyInference:
    def __init__(self, model_dir: str, config_path: Path = CONFIG_PATH) -> None:
        col_map = _load_column_map(config_path)
        self._feature_keys = list(col_map["features"].keys())

        model_path = Path(model_dir) / "anomaly_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

        self._clf = joblib.load(model_path)
        self.model_name = "isolation_forest_anomaly"
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def predict(self, input_data: dict) -> dict:
        """Input/output per MASTER_SPEC §3.5."""
        t_start = time.perf_counter()

        sensors = input_data["sensors"]
        X = np.array([[sensors[k] for k in self._feature_keys]])

        pred          = int(self._clf.predict(X)[0])
        is_anomaly    = pred == -1
        raw_score     = float(self._clf.decision_function(X)[0])
        anomaly_score = round(_sigmoid(raw_score * 5), 4)

        return {
            "model":             self.model_name,
            "node_id":           input_data["node_id"],
            "timestamp":         input_data["timestamp"],
            "is_anomaly":        is_anomaly,
            "anomaly_score":     anomaly_score,
            "inference_time_ms": round((time.perf_counter() - t_start) * 1000, 3),
        }
