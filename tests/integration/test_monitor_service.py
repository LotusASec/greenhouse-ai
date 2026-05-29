"""Integration tests for Phase 2C — Model Output Monitor service.

Requires the service running at localhost:8105 with DB_PATH set.
Start before running:
    cd models/output_monitor
    DB_PATH=/tmp/greenhouse_monitor_test.db NODE_ID=greenhouse_01 \\
        python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8105

Run:
    pytest tests/integration/test_monitor_service.py -v
"""

import os
import sqlite3
import time
from pathlib import Path

import requests

MONITOR_URL = "http://localhost:8105"
NODE_ID     = "greenhouse_01"
TIMESTAMP   = "2024-01-01T12:00:00Z"
DB_PATH     = os.getenv("DB_PATH", "/tmp/greenhouse_monitor_test.db")


def _post(model_name: str, metric_value: float) -> requests.Response:
    return requests.post(
        f"{MONITOR_URL}/monitor/{model_name}",
        json={
            "node_id":      NODE_ID,
            "timestamp":    TIMESTAMP,
            "metric_value": metric_value,
        },
        timeout=10,
    )


# ---------------------------------------------------------------------------
# 1. test_monitor_health
# ---------------------------------------------------------------------------

def test_monitor_health():
    """GET /health → 200, status: ok."""
    resp = requests.get(f"{MONITOR_URL}/health", timeout=5)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert data["status"] == "ok", f"status not ok: {data}"
    assert "service" in data


# ---------------------------------------------------------------------------
# 2. test_monitor_normal_flow
# ---------------------------------------------------------------------------

def test_monitor_normal_flow():
    """POST /monitor/disease 30 times with confidence ~0.88 → buffer_size 30, warming_up false."""
    # Fresh model state
    for _ in range(30):
        resp = _post("disease", 0.88)
        assert resp.status_code == 200, f"Unexpected status {resp.status_code}: {resp.text}"
        data = resp.json()
        # Schema check — MASTER_SPEC §3.6
        assert "node_id"       in data
        assert "timestamp"     in data
        assert "model_name"    in data
        assert "metric_value"  in data
        assert "z_score"       in data
        assert "window_mean"   in data
        assert "window_std"    in data
        assert "is_anomaly"    in data
        assert isinstance(data["is_anomaly"], bool)

    status_resp = requests.get(f"{MONITOR_URL}/status/disease", timeout=5)
    assert status_resp.status_code == 200
    status = status_resp.json()
    assert status["buffer_size"] >= 10, (
        f"Expected buffer_size >= 10 after 30 POSTs, got {status['buffer_size']}"
    )
    assert status["warming_up"] is False, (
        f"Expected warming_up=False after 30 samples, got {status}"
    )


# ---------------------------------------------------------------------------
# 3. test_monitor_drift_detection
# ---------------------------------------------------------------------------

def test_monitor_drift_detection():
    """Fill nutrition with 40 normal values, then send spike 0.10 → is_anomaly: true."""
    # Seed with tightly-clustered normal values
    for _ in range(40):
        resp = _post("nutrition", 0.88)
        assert resp.status_code == 200

    # Clear spike
    spike_resp = _post("nutrition", 0.10)
    assert spike_resp.status_code == 200
    data = spike_resp.json()
    assert data["is_anomaly"] is True, (
        f"Expected is_anomaly=True for spike 0.10, got {data}"
    )


# ---------------------------------------------------------------------------
# 4. test_monitor_all_models
# ---------------------------------------------------------------------------

def test_monitor_all_models():
    """POST to all 4 model endpoints → GET /status shows all 4 models."""
    for model in ("disease", "irrigation", "nutrition", "anomaly"):
        resp = _post(model, 0.88)
        assert resp.status_code == 200, (
            f"POST /monitor/{model} returned {resp.status_code}"
        )
        data = resp.json()
        assert data["model_name"] == model

    status = requests.get(f"{MONITOR_URL}/status", timeout=5).json()
    for model in ("disease", "irrigation", "nutrition", "anomaly"):
        assert model in status, f"Model '{model}' missing from /status"
        assert "buffer_size" in status[model]
        assert "warming_up"  in status[model]


# ---------------------------------------------------------------------------
# 5. test_sqlite_write
# ---------------------------------------------------------------------------

def test_sqlite_write():
    """POST /monitor/disease → row appears in SQLite monitor_events with correct node_id."""
    if not Path(DB_PATH).exists():
        import pytest
        pytest.skip(f"DB_PATH={DB_PATH} not found — start service with DB_PATH set")

    # Count rows before
    conn   = sqlite3.connect(DB_PATH)
    before = conn.execute("SELECT COUNT(*) FROM monitor_events").fetchone()[0]
    conn.close()

    # POST one event
    resp = _post("disease", 0.91)
    assert resp.status_code == 200

    # Short wait for write
    time.sleep(0.1)

    conn  = sqlite3.connect(DB_PATH)
    after = conn.execute("SELECT COUNT(*) FROM monitor_events").fetchone()[0]
    row   = conn.execute(
        "SELECT node_id, model_name FROM monitor_events ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    assert after > before, "No new row written to monitor_events"
    assert row is not None
    assert row[0] == NODE_ID,   f"node_id mismatch: {row[0]} != {NODE_ID}"
    assert row[1] == "disease", f"model_name mismatch: {row[1]} != disease"
