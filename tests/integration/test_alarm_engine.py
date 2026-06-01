"""Integration tests for Alarm Engine API — Phase 4, T4.4.

7 tests against the running service on port 8107.
Requires: docker compose up edge1_alarm (or local uvicorn).

Run: pytest tests/integration/test_alarm_engine.py -v
"""

import time
import uuid

import pytest
import requests

BASE = "http://localhost:8107"


def make_alarm(
    alarm_id: str | None = None,
    node_id: str = "greenhouse_01",
    level: str = "WARNING",
    source: str = "rule_engine",
    rule_id: str | None = "RULE_002",
) -> dict:
    return {
        "alarm_id": alarm_id or f"ALM_20240115_{uuid.uuid4().hex[:6].upper()}",
        "node_id": node_id,
        "timestamp": "2024-01-15T10:00:00",
        "level": level,
        "source": source,
        "rule_id": rule_id,
        "trigger_values": {"sensor_reading.sensors.soil_moisture": 25.0},
        "llm_explanation": None,
        "synced": False,
    }


def check_service_available():
    try:
        r = requests.get(f"{BASE}/health", timeout=3)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


@pytest.fixture(autouse=True)
def require_service():
    if not check_service_available():
        pytest.skip("edge1_alarm not running on port 8107 — start with docker compose up edge1_alarm")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_alarm_health():
    """GET /health returns 200 with status ok."""
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "cooldown_seconds" in data


def test_post_alarm_stored():
    """POST /alarm stores in SQLite and GET /alerts returns it."""
    alarm = make_alarm()
    r = requests.post(f"{BASE}/alarm", json=alarm)
    assert r.status_code == 200
    result = r.json()
    assert result["stored"] is True
    assert result["suppressed"] is False

    alerts = requests.get(f"{BASE}/alerts").json()
    ids = [a["alarm_id"] for a in alerts]
    assert alarm["alarm_id"] in ids


def test_post_alarm_suppressed():
    """Posting the same rule_id twice within 60s suppresses the second."""
    rule_id = f"RULE_INT_{uuid.uuid4().hex[:4].upper()}"
    alarm1 = make_alarm(rule_id=rule_id)
    alarm2 = make_alarm(alarm_id=f"ALM_20240115_{uuid.uuid4().hex[:6].upper()}", rule_id=rule_id)

    r1 = requests.post(f"{BASE}/alarm", json=alarm1)
    r2 = requests.post(f"{BASE}/alarm", json=alarm2)

    assert r1.status_code == 200
    assert r2.status_code == 200

    result1 = r1.json()
    result2 = r2.json()

    assert result1["stored"] is True
    assert result2["suppressed"] is True

    # Only the first alarm should appear in the DB
    alerts = requests.get(f"{BASE}/alerts").json()
    ids = [a["alarm_id"] for a in alerts]
    assert alarm1["alarm_id"] in ids
    assert alarm2["alarm_id"] not in ids


def test_filter_critical():
    """GET /alerts?level=CRITICAL returns only CRITICAL alarms."""
    crit_id = f"ALM_20240115_{uuid.uuid4().hex[:6].upper()}"
    warn_id = f"ALM_20240115_{uuid.uuid4().hex[:6].upper()}"

    requests.post(f"{BASE}/alarm", json=make_alarm(alarm_id=crit_id, level="CRITICAL", rule_id=f"RULE_C_{uuid.uuid4().hex[:4]}"))
    requests.post(f"{BASE}/alarm", json=make_alarm(alarm_id=warn_id, level="WARNING", rule_id=f"RULE_W_{uuid.uuid4().hex[:4]}"))

    r = requests.get(f"{BASE}/alerts", params={"level": "CRITICAL"})
    assert r.status_code == 200
    alarms = r.json()
    assert all(a["level"] == "CRITICAL" for a in alarms)
    assert any(a["alarm_id"] == crit_id for a in alarms)


def test_summary_endpoint():
    """GET /alerts/summary returns total, by_level, by_source, unsynced."""
    r = requests.get(f"{BASE}/alerts/summary")
    assert r.status_code == 200
    summary = r.json()
    assert "total" in summary
    assert "by_level" in summary
    assert "by_source" in summary
    assert "unsynced" in summary
    by_level = summary["by_level"]
    for key in ("CRITICAL", "HIGH", "WARNING", "INFO"):
        assert key in by_level


def test_sync_flow():
    """POST /alarm → in unsynced → POST sync → removed from unsynced."""
    rule_id = f"RULE_SYNC_{uuid.uuid4().hex[:4].upper()}"
    alarm = make_alarm(rule_id=rule_id)
    r = requests.post(f"{BASE}/alarm", json=alarm)
    assert r.json()["stored"] is True
    alarm_id = alarm["alarm_id"]

    unsynced_before = requests.get(f"{BASE}/alerts/unsynced").json()
    ids_before = [a["alarm_id"] for a in unsynced_before["alarms"]]
    assert alarm_id in ids_before

    sync_r = requests.post(f"{BASE}/alerts/{alarm_id}/sync")
    assert sync_r.status_code == 200
    assert sync_r.json()["updated"] is True

    unsynced_after = requests.get(f"{BASE}/alerts/unsynced").json()
    ids_after = [a["alarm_id"] for a in unsynced_after["alarms"]]
    assert alarm_id not in ids_after


def test_pagination():
    """GET /alerts?limit=2&offset=N returns correct non-overlapping pages."""
    posted_ids = []
    for _ in range(5):
        alarm = make_alarm(rule_id=None, source="model_monitor")
        requests.post(f"{BASE}/alarm", json=alarm)
        posted_ids.append(alarm["alarm_id"])

    page0 = requests.get(f"{BASE}/alerts", params={"limit": 2, "offset": 0}).json()
    page1 = requests.get(f"{BASE}/alerts", params={"limit": 2, "offset": 2}).json()

    assert len(page0) == 2
    assert len(page1) == 2

    ids0 = {a["alarm_id"] for a in page0}
    ids1 = {a["alarm_id"] for a in page1}
    assert ids0.isdisjoint(ids1)
