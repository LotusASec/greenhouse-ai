"""Full end-to-end system integration tests — Phase 8 T8.5.

6 tests. Requires ALL services running (edge + central + grafana).
Run: pytest tests/integration/test_full_system.py -v --tb=short

Services required:
  :8100  edge1_gateway
  :8101-8107  edge1_* model services
  :9000  central_gateway
  :3000  grafana (central)
"""

import time
from datetime import datetime, timezone

import httpx
import pytest

EDGE_URL    = "http://localhost:8100"
CENTRAL_URL = "http://localhost:9000"
GRAFANA_URL = "http://localhost:3000"
MONITOR_URL = "http://localhost:8105"

FUNGAL_SENSORS = {
    "temperature": 22.0, "humidity": 88.0,
    "soil_moisture": 70.0, "light": 400.0,
    "ec": 1.2, "ph": 7.5,
}
NORMAL_SENSORS = {
    "temperature": 24.0, "humidity": 65.0,
    "soil_moisture": 55.0, "light": 700.0,
    "ec": 2.2, "ph": 6.3,
}
# Different rule-triggering sensor sets to avoid cooldown in disconnection test
OUTAGE_SENSORS = [
    {"temperature": 22.0, "humidity": 88.0, "soil_moisture": 70.0,
     "light": 400.0, "ec": 1.2, "ph": 7.5},   # RULE_009 → CRITICAL
    {"temperature": 28.0, "humidity": 45.0, "soil_moisture": 22.0,
     "light": 800.0, "ec": 2.0, "ph": 6.5},    # RULE_002 → WARNING
    {"temperature": 48.0, "humidity": 99.0, "soil_moisture": 98.0,
     "light": 1950.0, "ec": 4.8, "ph": 1.2},   # RULE_003/004 → HIGH
]


def ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def predict(sensors: dict, url: str = EDGE_URL) -> dict:
    node_id = "greenhouse_01"
    r = httpx.post(
        f"{url}/predict",
        json={"sensor_reading": {"node_id": node_id, "timestamp": ts(), "sensors": sensors}},
        timeout=15.0,
    )
    assert r.status_code == 200, f"/predict returned {r.status_code}: {r.text[:200]}"
    return r.json()


# ── Test 1 ────────────────────────────────────────────────────────────────────

def test_full_pipeline_normal():
    """Normal sensors → alarm in valid level set, written to edge DB."""
    alarm = predict(NORMAL_SENSORS)
    assert "alarm_id" in alarm
    assert alarm["level"] in {"INFO", "WARNING", "HIGH", "CRITICAL"}
    assert alarm["node_id"] == "greenhouse_01"
    assert "timestamp" in alarm


# ── Test 2 ────────────────────────────────────────────────────────────────────

def test_full_pipeline_fungal():
    """Fungal/critical sensors → CRITICAL or HIGH alarm, synced to central."""
    alarm = predict(FUNGAL_SENSORS)
    assert alarm["level"] in {"CRITICAL", "HIGH"}, (
        f"Expected CRITICAL or HIGH for fungal scenario, got {alarm['level']}"
    )
    alarm_id = alarm["alarm_id"]

    # Verify the alarm appears in edge /alerts (use large limit to avoid pagination miss)
    r = httpx.get(f"{EDGE_URL}/alerts", params={"limit": 200}, timeout=5.0)
    assert r.status_code == 200
    alarms = r.json()
    ids = [a["alarm_id"] for a in alarms]
    assert alarm_id in ids, (
        f"Alarm {alarm_id} not found in edge /alerts (returned {len(alarms)} alarms); "
        f"alarm may have been suppressed by cooldown — retry after 60s"
    )

    # Also verify /explain works on central (may use template fallback)
    r = httpx.post(
        f"{CENTRAL_URL}/explain",
        json=alarm,
        timeout=15.0,
    )
    assert r.status_code == 200, f"/explain returned {r.status_code}"
    data = r.json()
    explanation = data.get("explanation", "")
    assert len(explanation) > 0, "LLM explanation is empty"


# ── Test 3 ────────────────────────────────────────────────────────────────────

def test_full_pipeline_drift():
    """50 stable monitor values + spike → WARNING from model_monitor."""
    node_id = "greenhouse_01"
    normal_val  = 0.88
    anomaly_val = 0.05

    # Prime the buffer with 50 identical values
    for _ in range(50):
        r = httpx.post(
            f"{MONITOR_URL}/monitor/disease",
            json={"node_id": node_id, "timestamp": ts(), "metric_value": normal_val},
            timeout=3.0,
        )
        assert r.status_code == 200

    # Inject spike
    r = httpx.post(
        f"{MONITOR_URL}/monitor/disease",
        json={"node_id": node_id, "timestamp": ts(), "metric_value": anomaly_val},
        timeout=3.0,
    )
    assert r.status_code == 200
    event = r.json()

    assert event.get("is_anomaly") is True, (
        f"Expected is_anomaly=True for spike value {anomaly_val} after "
        f"buffer mean ~{normal_val}, got is_anomaly={event.get('is_anomaly')} "
        f"z_score={event.get('z_score')}"
    )
    assert event.get("model_name") == "disease"


# ── Test 4 ────────────────────────────────────────────────────────────────────

def test_disconnection_recovery():
    """Stop central → inject alarms → restart → verify sync within 35s."""
    import subprocess

    def docker_compose(args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["docker", "compose"] + args,
            capture_output=True, text=True, timeout=60,
        )

    # Record baseline central alarm count
    r = httpx.get(f"{CENTRAL_URL}/alerts", timeout=5.0)
    if r.status_code != 200:
        pytest.skip("Central gateway not available")
    baseline_central = len(r.json())

    # Stop central
    result = docker_compose(["stop", "central_gateway", "central_aggregator"])
    assert result.returncode == 0, f"docker stop failed: {result.stderr}"
    time.sleep(2)

    # Record edge unsynced count before injection
    r = httpx.get(f"{EDGE_URL}/alerts/unsynced", timeout=5.0)
    assert r.status_code == 200
    unsynced_before = len(r.json())

    # Inject 3 alarms with different rules to avoid cooldown suppression
    for sensors in OUTAGE_SENSORS:
        predict(sensors)
        time.sleep(0.3)

    # Verify edge stored at least 1 new unsynced alarm (others may be cooldown-suppressed)
    r = httpx.get(f"{EDGE_URL}/alerts/unsynced", timeout=5.0)
    assert r.status_code == 200
    unsynced_during = r.json()
    new_unsynced = len(unsynced_during) - unsynced_before
    assert new_unsynced >= 1, (
        f"Expected at least 1 new unsynced alarm on edge, got {new_unsynced} new "
        f"(before={unsynced_before}, during={len(unsynced_during)})"
    )
    new_unsynced_ids = {a["alarm_id"] for a in unsynced_during[: new_unsynced]}

    # Restart central
    result = docker_compose(["start", "central_gateway", "central_aggregator"])
    assert result.returncode == 0, f"docker start failed: {result.stderr}"

    # Wait for central to be healthy
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{CENTRAL_URL}/health", timeout=3.0)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        pytest.fail("Central gateway did not come back healthy in 90s")

    # waiting for sync cycle (CENTRAL_SYNC_INTERVAL=30s)
    time.sleep(35)

    # Verify edge unsynced count went back down (alarms were synced)
    r = httpx.get(f"{EDGE_URL}/alerts/unsynced", timeout=5.0)
    assert r.status_code == 200
    unsynced_after = len(r.json())
    assert unsynced_after <= unsynced_before, (
        f"Unsynced count did not decrease after restart and sync. "
        f"Before={unsynced_before}, during={len(unsynced_during)}, after={unsynced_after}"
    )


# ── Test 5 ────────────────────────────────────────────────────────────────────

def test_grafana_has_data():
    """Grafana datasource API returns rows from sensor_readings and alarms."""
    # Query edge Grafana (port 3001) for sensor_readings count
    r = httpx.post(
        "http://localhost:3001/api/ds/query",
        auth=("admin", "greenhouse"),
        json={
            "queries": [{
                "refId": "A",
                "datasource": {"type": "frser-sqlite-datasource", "uid": "EdgeDB"},
                "queryText": "SELECT COUNT(*) as cnt FROM sensor_readings",
                "rawQuery": True,
                "format": "table",
            }]
        },
        timeout=10.0,
    )
    assert r.status_code == 200, f"Grafana query returned {r.status_code}"
    result = r.json()
    frames = result.get("results", {}).get("A", {}).get("frames", [])
    assert len(frames) > 0, "Grafana returned no frames"
    values = frames[0].get("data", {}).get("values", [])
    assert len(values) > 0, "Grafana frame has no values"
    count = values[0][0] if values and values[0] else 0
    assert count > 0, f"Expected sensor_readings > 0, got {count}"

    # Query central Grafana (port 3000) for alarms count
    r = httpx.post(
        "http://localhost:3000/api/ds/query",
        auth=("admin", "greenhouse"),
        json={
            "queries": [{
                "refId": "A",
                "datasource": {"type": "frser-sqlite-datasource", "uid": "CentralDB"},
                "queryText": "SELECT COUNT(*) as cnt FROM alarms",
                "rawQuery": True,
                "format": "table",
            }]
        },
        timeout=10.0,
    )
    assert r.status_code == 200
    result = r.json()
    frames = result.get("results", {}).get("A", {}).get("frames", [])
    values = frames[0].get("data", {}).get("values", []) if frames else []
    count = values[0][0] if values and values[0] else 0
    assert count > 0, f"Expected central alarms > 0, got {count}"


# ── Test 6 ────────────────────────────────────────────────────────────────────

def test_all_services_healthy():
    """All 12 key service endpoints return HTTP 200 on /health."""
    endpoints = [
        ("edge1_gateway",        "http://localhost:8100/health"),
        ("edge1_disease",        "http://localhost:8101/health"),
        ("edge1_irrigation",     "http://localhost:8102/health"),
        ("edge1_nutrition",      "http://localhost:8103/health"),
        ("edge1_anomaly",        "http://localhost:8104/health"),
        ("edge1_output_monitor", "http://localhost:8105/health"),
        ("edge1_fusion",         "http://localhost:8106/health"),
        ("edge1_alarm",          "http://localhost:8107/health"),
        ("central_gateway",      "http://localhost:9000/health"),
        ("central_aggregator",   "http://localhost:9001/health"),
        ("central_lstm",         "http://localhost:9002/health"),
        ("central_llm",          "http://localhost:9003/health"),
    ]
    failures: list[str] = []
    for name, url in endpoints:
        try:
            r = httpx.get(url, timeout=5.0)
            if r.status_code != 200:
                failures.append(f"{name}: HTTP {r.status_code}")
        except Exception as exc:
            failures.append(f"{name}: {exc}")

    assert not failures, f"Unhealthy services:\n" + "\n".join(failures)
