"""Unit tests for MonitorService — T2C.3 (Phase 2C).

Run:
    pytest tests/unit/test_monitor.py -v

All 6 must pass before T2C.2 (API) is implemented.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add output_monitor directory so 'monitor' is importable
sys.path.insert(0, str(Path(__file__).parents[2] / "models" / "output_monitor"))

from monitor import MonitorService

NODE_ID   = "greenhouse_01"
TIMESTAMP = "2024-01-01T12:00:00Z"


def _check(svc: MonitorService, model: str, value: float) -> dict:
    return svc.check(model, value, NODE_ID, TIMESTAMP)


# ---------------------------------------------------------------------------
# 1. test_warming_up
# ---------------------------------------------------------------------------

def test_warming_up():
    """9 values sent → all is_anomaly: false (buffer < MIN_SAMPLES=10)."""
    svc = MonitorService()
    for _ in range(9):
        result = _check(svc, "disease", 0.85)
        assert result["is_anomaly"] is False, (
            f"Expected is_anomaly=False during warm-up, got {result}"
        )
        assert result["z_score"] == 0.0
        assert result["window_std"] == 0.0


# ---------------------------------------------------------------------------
# 2. test_normal_values
# ---------------------------------------------------------------------------

def test_normal_values():
    """50 values from N(0.85, 0.05) → false positive rate < 5%."""
    svc = MonitorService()
    rng = np.random.default_rng(42)
    values = rng.normal(0.85, 0.05, 50).clip(0.0, 1.0)

    anomalies = 0
    for v in values:
        result = _check(svc, "irrigation", float(v))
        if result["is_anomaly"]:
            anomalies += 1

    rate = anomalies / 50
    assert rate < 0.05, (
        f"False positive rate {rate:.2f} exceeds 5% on normal N(0.85,0.05) data"
    )


# ---------------------------------------------------------------------------
# 3. test_spike_detection
# ---------------------------------------------------------------------------

def test_spike_detection():
    """Fill with 40 normal N(0.85, 0.03) values, send spike 0.10 → is_anomaly: true."""
    svc = MonitorService()
    rng = np.random.default_rng(0)

    for v in rng.normal(0.85, 0.03, 40).clip(0.0, 1.0):
        _check(svc, "disease", float(v))

    spike_result = _check(svc, "disease", 0.10)
    assert spike_result["is_anomaly"] is True, (
        f"Expected is_anomaly=True for spike 0.10, got {spike_result}"
    )
    assert abs(spike_result["z_score"]) > 2.5, (
        f"|z_score| = {abs(spike_result['z_score']):.4f}, expected > 2.5"
    )


# ---------------------------------------------------------------------------
# 4. test_freeze_detection
# ---------------------------------------------------------------------------

def test_freeze_detection():
    """All values identical (std≈0) → no crash, z-score protected by 1e-9 guard."""
    svc = MonitorService()

    # Fill buffer with a constant value
    for _ in range(15):
        _check(svc, "nutrition", 0.85)

    # std is ~0 here; z-score should be 0 or near-zero (not NaN/Inf)
    result = _check(svc, "nutrition", 0.85)

    assert isinstance(result["is_anomaly"], bool), "is_anomaly must be bool"
    assert isinstance(result["z_score"], float), "z_score must be float"
    assert not (result["z_score"] != result["z_score"]), "z_score must not be NaN"
    assert abs(result["z_score"]) < 1e6, "z_score must not explode (std=0 guard)"

    # Sending the same value as the buffer mean → z ≈ 0, not an anomaly
    assert result["is_anomaly"] is False, (
        "Freeze (same value as mean) should not trigger anomaly"
    )


# ---------------------------------------------------------------------------
# 5. test_independent_buffers
# ---------------------------------------------------------------------------

def test_independent_buffers():
    """Spike in disease buffer must not affect irrigation buffer."""
    svc = MonitorService()
    rng = np.random.default_rng(1)

    # Fill both buffers with identical normal data
    for v in rng.normal(0.85, 0.03, 40).clip(0.0, 1.0):
        _check(svc, "disease",    float(v))
        _check(svc, "irrigation", float(v))

    # Inject spike into disease only
    disease_result    = _check(svc, "disease",    0.10)
    irrigation_result = _check(svc, "irrigation", 0.85)

    assert disease_result["is_anomaly"] is True, (
        "Spike 0.10 must be detected as anomaly in disease buffer"
    )
    assert irrigation_result["is_anomaly"] is False, (
        "Normal value in irrigation buffer must not be affected by disease spike"
    )


# ---------------------------------------------------------------------------
# 6. test_buffer_rollover
# ---------------------------------------------------------------------------

def test_buffer_rollover():
    """Send 60 values (> WINDOW_SIZE=50) → buffer stays capped at 50."""
    svc = MonitorService()

    for _ in range(60):
        _check(svc, "anomaly", 0.5)

    status = svc.get_status("anomaly")
    assert status["buffer_size"] == 50, (
        f"Expected buffer_size=50 after 60 inserts, got {status['buffer_size']}"
    )
    assert status["total_events"] == 60, (
        f"Expected total_events=60, got {status['total_events']}"
    )
    assert status["warming_up"] is False
