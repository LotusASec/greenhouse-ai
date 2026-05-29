"""
Integration smoke tests for Phase 2A sensor models.

Requires services running at localhost:8102, 8103, 8104.
Start before running:
    MODEL_DIR=model_weights uvicorn api.main:app --port 8102  (irrigation dir)
    MODEL_DIR=model_weights uvicorn api.main:app --port 8103  (nutrition dir)
    MODEL_DIR=model_weights uvicorn api.main:app --port 8104  (anomaly dir)

Run:
    pytest tests/integration/test_sensor_models.py -v
"""

import sys
from pathlib import Path

import pytest
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IRRIGATION_URL = "http://localhost:8102"
NUTRITION_URL  = "http://localhost:8103"
ANOMALY_URL    = "http://localhost:8104"

NORMAL_PAYLOAD = {
    "node_id":   "greenhouse_01",
    "timestamp": "2024-01-01T12:00:00Z",
    "sensors": {
        "temperature":   24.0,
        "humidity":      65.0,
        "soil_moisture": 55.0,
        "light":         700.0,
        "ec":            2.2,
        "ph":            6.3,
    },
}

EXTREME_PAYLOAD = {
    "node_id":   "greenhouse_01",
    "timestamp": "2024-01-01T12:00:00Z",
    "sensors": {
        "temperature":   100.0,
        "humidity":      200.0,
        "soil_moisture": 150.0,
        "light":         5000.0,
        "ec":            15.0,
        "ph":            14.0,
    },
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _post(url: str, payload: dict) -> requests.Response:
    return requests.post(url + "/predict", json=payload, timeout=10)


# ---------------------------------------------------------------------------
# 1. Simulator output bounds
# ---------------------------------------------------------------------------

def test_simulator_output_bounds():
    """Generate 100 readings from node1 config; assert all within yaml min/max."""
    sim_dir = Path(__file__).parents[2] / "simulator"
    sys.path.insert(0, str(sim_dir))
    from sensor_simulator import SensorSimulator
    import yaml

    cfg_path = sim_dir / "config" / "node1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    profile = cfg["sensor_profile"]

    sim = SensorSimulator(config_path=str(cfg_path), db_path=None)

    violations = []
    for _ in range(100):
        reading = sim.generate_reading()
        for sensor, value in reading["sensors"].items():
            lo = float(profile[sensor]["min"])
            hi = float(profile[sensor]["max"])
            if not (lo <= value <= hi):
                violations.append(f"{sensor}={value:.3f} outside [{lo},{hi}]")

    assert len(violations) == 0, f"Bound violations: {violations}"


# ---------------------------------------------------------------------------
# 2. Irrigation predict
# ---------------------------------------------------------------------------

def test_irrigation_predict():
    """POST to irrigation service; assert MASTER_SPEC §3.3 schema."""
    resp = _post(IRRIGATION_URL, NORMAL_PAYLOAD)
    assert resp.status_code == 200, f"Unexpected status {resp.status_code}"

    data = resp.json()
    assert data["model"]             == "xgboost_irrigation"
    assert isinstance(data["node_id"],           str)
    assert isinstance(data["timestamp"],         str)
    assert isinstance(data["irrigate"],          bool)
    assert isinstance(data["amount_liters"],     float)
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["inference_time_ms"], float)
    assert data["amount_liters"] >= 0.0


# ---------------------------------------------------------------------------
# 3. Nutrition predict
# ---------------------------------------------------------------------------

def test_nutrition_predict():
    """POST to nutrition service; assert MASTER_SPEC §3.4 schema."""
    resp = _post(NUTRITION_URL, NORMAL_PAYLOAD)
    assert resp.status_code == 200, f"Unexpected status {resp.status_code}"

    data = resp.json()
    assert data["model"]          == "rf_nutrition"
    assert isinstance(data["node_id"],                   str)
    assert isinstance(data["timestamp"],                 str)
    assert data["deficiency_class"] in {
        "N_deficiency", "P_deficiency", "K_deficiency", "normal"
    }
    assert isinstance(data["fertilizer_recommendation"], str)
    assert len(data["fertilizer_recommendation"]) > 0
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["inference_time_ms"],         float)


# ---------------------------------------------------------------------------
# 4. Anomaly predict — normal values
# ---------------------------------------------------------------------------

def test_anomaly_predict_normal():
    """Normal sensor values → is_anomaly: false."""
    resp = _post(ANOMALY_URL, NORMAL_PAYLOAD)
    assert resp.status_code == 200, f"Unexpected status {resp.status_code}"

    data = resp.json()
    assert data["model"]    == "isolation_forest_anomaly"
    assert data["is_anomaly"] is False
    assert isinstance(data["anomaly_score"],     float)
    assert isinstance(data["inference_time_ms"], float)


# ---------------------------------------------------------------------------
# 5. Anomaly predict — extreme values
# ---------------------------------------------------------------------------

def test_anomaly_predict_extreme():
    """Clearly extreme sensor values → is_anomaly: true."""
    resp = _post(ANOMALY_URL, EXTREME_PAYLOAD)
    assert resp.status_code == 200, f"Unexpected status {resp.status_code}"

    data = resp.json()
    assert data["is_anomaly"] is True, (
        f"Expected is_anomaly=true for extreme sensors, got {data}"
    )
    assert isinstance(data["anomaly_score"], float)


# ---------------------------------------------------------------------------
# 6. Health endpoints
# ---------------------------------------------------------------------------

def test_all_health_endpoints():
    """All three model services respond to GET /health with status: ok."""
    for url, name in [
        (IRRIGATION_URL, "irrigation"),
        (NUTRITION_URL,  "nutrition"),
        (ANOMALY_URL,    "anomaly"),
    ]:
        resp = requests.get(url + "/health", timeout=5)
        assert resp.status_code == 200, f"{name} health returned {resp.status_code}"
        data = resp.json()
        assert data["status"] == "ok", f"{name} health status: {data}"
