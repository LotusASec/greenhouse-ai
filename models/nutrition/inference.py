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

        # 1. Ham tahmini sayısal (int) olarak al
        raw_pred = self._clf.predict(X)[0]  # Örn: 1 (int)
        proba    = self._clf.predict_proba(X)[0]

        # 2. Olasılık indexini sayısal değer üzerinden güvenle bul
        class_idx = list(self._clf.classes_).index(raw_pred)
        confidence = float(proba[class_idx])

        # 3. Sayısal sınıfı MASTER_SPEC §3.4 şemasına uygun string etiketine çevir
        class_mapping = {
            0: "N_deficiency",
            1: "P_deficiency",
            2: "K_deficiency",
            3: "normal"
        }
        
        # Eğer modelden gelen değer int veya string sayı ise maple, yoksa olduğu gibi bırak
        try:
            deficiency_class = class_mapping[int(raw_pred)]
        except (ValueError, KeyError):
            deficiency_class = str(raw_pred)

        recommendation = FERTILIZER_RECOMMENDATIONS.get(
            deficiency_class, "Agronomik değerlendirme yapın."
        )

        t_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "model": "rf_nutrition",
            "node_id": input_data.get("node_id", "unknown"),
            "timestamp": input_data.get("timestamp", ""),
            "deficiency_class": deficiency_class,
            "confidence": confidence,
            "fertilizer_recommendation": recommendation,
            "inference_time_ms": t_ms
        }

        