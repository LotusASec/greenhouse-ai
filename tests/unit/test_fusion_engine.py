"""Unit tests for FusionEngine and RuleEngine — Phase 3, T3.4.

10 tests covering all rules, monitor events, merge logic,
alarm_id format, nested field resolution, and validation.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

# Load from exact path so pytest can run all unit tests together without
# module-name collision against edge/alarm_engine/engine.py
_ENGINE_PATH = (
    Path(__file__).parent.parent.parent / "edge" / "fusion_engine" / "engine.py"
)
_spec = importlib.util.spec_from_file_location("fusion_engine_module", _ENGINE_PATH)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FusionEngine = _mod.FusionEngine
RuleEngine   = _mod.RuleEngine

RULES_PATH = str(
    Path(__file__).parent.parent.parent
    / "edge" / "fusion_engine" / "rules" / "rules.yaml"
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_fusion_input(
    *,
    humidity: float = 60.0,
    temperature: float = 24.0,
    soil_moisture: float = 50.0,
    light: float = 800.0,
    ec: float = 2.0,
    ph: float = 6.5,
    disease_prediction: str = "healthy",
    disease_confidence: float = 0.10,
    is_anomaly: bool = False,
    irrigate: bool = False,
    amount_liters: float = 0.0,
    deficiency_class: str = "normal",
    monitor_events: list | None = None,
) -> dict:
    return {
        "node_id": "greenhouse_01",
        "timestamp": "2024-01-15T10:00:00",
        "sensor_reading": {
            "node_id": "greenhouse_01",
            "timestamp": "2024-01-15T10:00:00",
            "sensors": {
                "temperature": temperature,
                "humidity": humidity,
                "soil_moisture": soil_moisture,
                "light": light,
                "ec": ec,
                "ph": ph,
            },
        },
        "disease_output": {
            "model": "resnet50_disease",
            "node_id": "greenhouse_01",
            "timestamp": "2024-01-15T10:00:00",
            "top_prediction": disease_prediction,
            "top_confidence": disease_confidence,
            "class_probabilities": {
                "healthy": 1.0 - disease_confidence,
                "early_blight": disease_confidence,
                "late_blight": 0.0,
                "leaf_mold": 0.0,
                "other": 0.0,
            },
            "inference_time_ms": 10.0,
        },
        "anomaly_output": {
            "model": "isolation_forest_anomaly",
            "node_id": "greenhouse_01",
            "timestamp": "2024-01-15T10:00:00",
            "is_anomaly": is_anomaly,
            "anomaly_score": -0.1 if is_anomaly else 0.1,
            "inference_time_ms": 2.0,
        },
        "irrigation_output": {
            "model": "xgboost_irrigation",
            "node_id": "greenhouse_01",
            "timestamp": "2024-01-15T10:00:00",
            "irrigate": irrigate,
            "amount_liters": amount_liters,
            "confidence": 0.95,
            "inference_time_ms": 3.0,
        },
        "nutrition_output": {
            "model": "rf_nutrition",
            "node_id": "greenhouse_01",
            "timestamp": "2024-01-15T10:00:00",
            "deficiency_class": deficiency_class,
            "fertilizer_recommendation": "none",
            "confidence": 0.95,
            "inference_time_ms": 2.0,
        },
        "monitor_events": monitor_events or [],
    }


def make_monitor_event(is_anomaly: bool = True, z_score: float = 3.1,
                       model_name: str = "resnet50_disease") -> dict:
    return {
        "node_id": "greenhouse_01",
        "timestamp": "2024-01-15T10:00:00",
        "model_name": model_name,
        "metric_value": 0.95,
        "z_score": z_score,
        "window_mean": 0.80,
        "window_std": 0.05,
        "is_anomaly": is_anomaly,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_rule_001_fungal():
    """RULE_001 fires: humidity>80, disease_confidence>0.85, is_anomaly=False → CRITICAL."""
    engine = FusionEngine(RULES_PATH)
    fi = make_fusion_input(
        humidity=85.0,
        disease_prediction="early_blight",
        disease_confidence=0.91,
        is_anomaly=False,
    )
    alarm = engine.process(fi)
    assert alarm["level"] == "CRITICAL"
    assert alarm["rule_id"] == "RULE_001"
    assert alarm["source"] == "rule_engine"


def test_rule_002_irrigation():
    """RULE_002 fires: soil_moisture<30, irrigate=True → WARNING."""
    engine = FusionEngine(RULES_PATH)
    fi = make_fusion_input(soil_moisture=25.0, irrigate=True)
    alarm = engine.process(fi)
    assert alarm["level"] == "WARNING"
    assert alarm["rule_id"] == "RULE_002"
    assert alarm["source"] == "rule_engine"


def test_no_rule_match():
    """All-normal input — RULE_010 (all_systems_normal) fires → INFO."""
    engine = FusionEngine(RULES_PATH)
    fi = make_fusion_input()
    alarm = engine.process(fi)
    assert alarm["level"] == "INFO"
    assert alarm["rule_id"] == "RULE_010"


def test_monitor_drift_warning():
    """Monitor event is_anomaly=True → WARNING from model_monitor source."""
    engine = FusionEngine(RULES_PATH)
    fi = make_fusion_input(monitor_events=[make_monitor_event(is_anomaly=True, z_score=3.1)])
    alarm = engine.process(fi)
    assert alarm["level"] == "WARNING"
    assert alarm["source"] == "model_monitor"


def test_rule_beats_monitor_when_higher():
    """Fungal conditions (CRITICAL) + monitor drift (WARNING) → CRITICAL from rule_engine."""
    engine = FusionEngine(RULES_PATH)
    fi = make_fusion_input(
        humidity=85.0,
        disease_prediction="early_blight",
        disease_confidence=0.91,
        is_anomaly=False,
        monitor_events=[make_monitor_event(is_anomaly=True, z_score=3.5)],
    )
    alarm = engine.process(fi)
    assert alarm["level"] == "CRITICAL"
    assert alarm["source"] == "rule_engine"


def test_monitor_beats_rule_when_higher():
    """Monitor WARNING beats rule INFO when no rule matches above INFO."""
    engine = FusionEngine(RULES_PATH)
    # Use inputs where NO named rule fires (anomaly=True so RULE_010 won't match,
    # but also no other rule fires): anomaly=True, temp<=32, disease=healthy, no irrigation
    fi = make_fusion_input(
        is_anomaly=True,
        temperature=28.0,  # not >32 so RULE_004 doesn't fire
        disease_prediction="healthy",
        disease_confidence=0.05,
        deficiency_class="normal",
        monitor_events=[make_monitor_event(is_anomaly=True, z_score=3.1)],
    )
    alarm = engine.process(fi)
    assert alarm["level"] == "WARNING"
    assert alarm["source"] == "model_monitor"


def test_alarm_id_format():
    """alarm_id must match ALM_YYYYMMDD_XXXXXX (6 uppercase hex chars)."""
    engine = FusionEngine(RULES_PATH)
    fi = make_fusion_input()
    alarm = engine.process(fi)
    assert re.match(r"^ALM_\d{8}_[A-F0-9]{6}$", alarm["alarm_id"]), (
        f"alarm_id format wrong: {alarm['alarm_id']}"
    )


def test_nested_field_resolution():
    """_get_nested resolves all nesting depths correctly."""
    engine = FusionEngine(RULES_PATH)
    re_engine = engine.rule_engine

    fi = make_fusion_input(humidity=75.5, disease_confidence=0.91, is_anomaly=False)

    # 3 levels: sensor_reading.sensors.humidity
    assert re_engine._get_nested(fi, "sensor_reading.sensors.humidity") == 75.5
    # 2 levels: disease_output.top_confidence
    assert re_engine._get_nested(fi, "disease_output.top_confidence") == 0.91
    # 2 levels: anomaly_output.is_anomaly
    assert re_engine._get_nested(fi, "anomaly_output.is_anomaly") is False
    # Missing field → None (no exception)
    assert re_engine._get_nested(fi, "sensor_reading.sensors.nonexistent") is None


def test_invalid_operator():
    """RuleEngine._evaluate_condition raises ValueError on unknown operator."""
    engine = FusionEngine(RULES_PATH)
    fi = make_fusion_input()
    cond = {"field": "sensor_reading.sensors.humidity", "operator": "invalid_op", "value": 50}
    with pytest.raises(ValueError, match="Unknown operator"):
        engine.rule_engine._evaluate_condition(cond, fi)


def test_rules_yaml_validation():
    """RuleEngine raises ValueError if a required rule field is missing."""
    import tempfile, os
    bad_yaml = """
rules:
  - id: "RULE_BAD"
    name: "bad_rule"
    conditions:
      - field: "sensor_reading.sensors.humidity"
        operator: "gt"
        value: 80
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(bad_yaml)
        tmp_path = f.name
    try:
        with pytest.raises(ValueError, match="action"):
            RuleEngine(tmp_path)
    finally:
        os.unlink(tmp_path)
