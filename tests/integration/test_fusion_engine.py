"""Integration tests for Fusion Engine service — Phase 3, T3.5.

Requires the fusion engine running at localhost:8106.
Start with:
    cd greenhouse-ai/edge/fusion_engine
    DB_PATH=/tmp/test_greenhouse.db uvicorn api.main:app --port 8106

Run:
    pytest tests/integration/test_fusion_engine.py -v
"""

import json
import os
import sqlite3
import time

import pytest
import requests

FUSION_URL = "http://localhost:8106"
DB_PATH = os.getenv("DB_PATH", "/tmp/test_greenhouse.db")


# ── Payload helpers ───────────────────────────────────────────────────────────

def base_payload(
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


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_fusion_health():
    """GET /health → 200, status ok."""
    r = requests.get(f"{FUSION_URL}/health", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["rules_loaded"] == 10


def test_fusion_normal_scenario():
    """POST /evaluate with all-normal values → INFO alarm, synced=False, alarm_id present."""
    r = requests.post(f"{FUSION_URL}/evaluate", json=base_payload(), timeout=5)
    assert r.status_code == 200
    alarm = r.json()
    assert alarm["level"] == "INFO"
    assert alarm["synced"] is False
    assert alarm["alarm_id"].startswith("ALM_")
    assert alarm["llm_explanation"] is None


def test_fusion_fungal_scenario():
    """POST /evaluate with fungal conditions → CRITICAL, source=rule_engine, rule_id=RULE_001."""
    payload = base_payload(
        humidity=85.0,
        disease_prediction="early_blight",
        disease_confidence=0.91,
        is_anomaly=False,
    )
    r = requests.post(f"{FUSION_URL}/evaluate", json=payload, timeout=5)
    assert r.status_code == 200
    alarm = r.json()
    assert alarm["level"] == "CRITICAL"
    assert alarm["source"] == "rule_engine"
    assert alarm["rule_id"] == "RULE_001"


def test_fusion_drift_scenario():
    """POST /evaluate with monitor drift event → WARNING, source=model_monitor."""
    monitor_event = {
        "node_id": "greenhouse_01",
        "timestamp": "2024-01-15T10:00:00",
        "model_name": "resnet50_disease",
        "metric_value": 0.95,
        "z_score": 3.1,
        "window_mean": 0.80,
        "window_std": 0.05,
        "is_anomaly": True,
    }
    # Use inputs where no rule fires above INFO:
    # anomaly=True but temp<=32 (RULE_004 needs temp>32), disease=healthy, no nutrition issues
    payload = base_payload(
        is_anomaly=True,
        temperature=28.0,
        disease_prediction="healthy",
        disease_confidence=0.05,
        deficiency_class="normal",
        monitor_events=[monitor_event],
    )
    r = requests.post(f"{FUSION_URL}/evaluate", json=payload, timeout=5)
    assert r.status_code == 200
    alarm = r.json()
    assert alarm["level"] == "WARNING"
    assert alarm["source"] == "model_monitor"


def test_sqlite_alarm_write():
    """POST /evaluate → alarm row written to SQLite alarms table with synced=0."""
    r = requests.post(f"{FUSION_URL}/evaluate", json=base_payload(), timeout=5)
    assert r.status_code == 200
    alarm_id = r.json()["alarm_id"]

    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT alarm_id, synced FROM alarms WHERE alarm_id = ?", (alarm_id,)
    ).fetchone()
    conn.close()

    assert row is not None, f"alarm_id {alarm_id} not found in SQLite"
    assert row[0] == alarm_id
    assert row[1] == 0  # synced = false


def test_get_rules():
    """GET /rules → list of 10 rules."""
    r = requests.get(f"{FUSION_URL}/rules", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 10
    assert len(data["rules"]) == 10
    rule_ids = [rule["id"] for rule in data["rules"]]
    assert "RULE_001" in rule_ids
    assert "RULE_010" in rule_ids


def test_threshold_update():
    """PUT /rules/RULE_001/threshold lowers humidity threshold; RULE_001 now fires at humidity=77."""
    # Get original threshold
    r = requests.get(f"{FUSION_URL}/rules", timeout=5)
    rules = r.json()["rules"]
    rule_001 = next(rule for rule in rules if rule["id"] == "RULE_001")
    original_val = next(
        c["value"] for c in rule_001["conditions"]
        if c["field"] == "sensor_reading.sensors.humidity"
    )

    # Lower threshold to 75
    r = requests.put(
        f"{FUSION_URL}/rules/RULE_001/threshold",
        json={"field": "sensor_reading.sensors.humidity", "value": 75.0},
        timeout=5,
    )
    assert r.status_code == 200

    # humidity=77 should now fire RULE_001
    payload = base_payload(
        humidity=77.0,
        disease_prediction="early_blight",
        disease_confidence=0.91,
        is_anomaly=False,
    )
    r = requests.post(f"{FUSION_URL}/evaluate", json=payload, timeout=5)
    assert r.status_code == 200
    alarm = r.json()
    assert alarm["level"] == "CRITICAL"
    assert alarm["rule_id"] == "RULE_001"

    # Restore original threshold
    requests.put(
        f"{FUSION_URL}/rules/RULE_001/threshold",
        json={"field": "sensor_reading.sensors.humidity", "value": original_val},
        timeout=5,
    )
