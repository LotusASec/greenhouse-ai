"""
Nutrition model inference — RandomForest multi-class classifier.

Feature order driven by config/column_map.yaml (TR-11, TR-17).
No column names are hardcoded in this file.
Output schema: MASTER_SPEC §3.4 (frozen)
"""

import time
from pathlib import Path

import joblib
import numpy as np
import yaml

CONFIG_PATH = Path(__file__).parent / "config" / "column_map.yaml"

FERTILIZER_RECOMMENDATIONS: dict[str, str] = {
    "N_deficiency": "Amonyum nitrat gübresi uygulayın. EC değerini 1.8-2.2 arasına çekin.",
    "P_deficiency": "Monoamonyum fosfat ekleyin. pH'ı 6.0-6.5 aralığına getirin.",
    "K_deficiency": "Potasyum sülfat uygulayın. EC ve pH dengesini kontrol edin.",
    "normal":       "Mevcut gübreleme programına devam edin.",
}


def _load_column_map(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class NutritionInference:
    def __init__(self, model_dir: str, config_path: Path = CONFIG_PATH) -> None:
        col_map = _load_column_map(config_path)
        # Feature order = key order in column_map.yaml (TR-11)
        self._feature_keys = list(col_map["features"].keys())

        model_path = Path(model_dir) / "nutrition_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

        self._clf = joblib.load(model_path)
        self.model_name = "rf_nutrition"
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def predict(self, input_data: dict) -> dict:
        """Input/output per MASTER_SPEC §3.4."""
        t_start = time.perf_counter()

        sensors = input_data["sensors"]
        X = np.array([[sensors[k] for k in self._feature_keys]])

        deficiency_class = str(self._clf.predict(X)[0])
        proba            = self._clf.predict_proba(X)[0]
        class_idx        = list(self._clf.classes_).index(deficiency_class)
        confidence       = float(proba[class_idx])
        recommendation   = FERTILIZER_RECOMMENDATIONS.get(
            deficiency_class, "Agronomik değerlendirme yapın."
        )

        return {
            "model":                     self.model_name,
            "node_id":                   input_data["node_id"],
            "timestamp":                 input_data["timestamp"],
            "deficiency_class":          deficiency_class,
            "fertilizer_recommendation": recommendation,
            "confidence":                round(confidence, 4),
            "inference_time_ms":         round((time.perf_counter() - t_start) * 1000, 3),
        }
