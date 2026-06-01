"""Unit tests for AlarmEngine — Phase 4, T4.3.

10 tests covering: storage, cooldown suppression, cooldown expiry,
monitor bypass, level/since filters, summary counts, sync marking,
pagination, and unknown alarm_id.

Uses mock datetime — never time.sleep().
"""

import importlib.util
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

# Load from exact path to avoid module-name collision with fusion_engine/engine.py
_ENGINE_PATH = (
    Path(__file__).parent.parent.parent / "edge" / "alarm_engine" / "engine.py"
)
_spec = importlib.util.spec_from_file_location("alarm_engine_module", _ENGINE_PATH)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["alarm_engine_module"] = _mod   # needed so patch() can find the module
_spec.loader.exec_module(_mod)
AlarmEngine = _mod.AlarmEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_alarm(
    alarm_id: str = "ALM_20240115_AABBCC",
    node_id: str = "greenhouse_01",
    timestamp: str = "2024-01-15T10:00:00",
    level: str = "WARNING",
    source: str = "rule_engine",
    rule_id: str | None = "RULE_002",
) -> dict:
    return {
        "alarm_id": alarm_id,
        "node_id": node_id,
        "timestamp": timestamp,
        "level": level,
        "source": source,
        "rule_id": rule_id,
        "trigger_values": {"sensor_reading.sensors.soil_moisture": 25.0},
        "llm_explanation": None,
        "synced": False,
    }


@pytest.fixture
def engine(tmp_path):
    db = str(tmp_path / "test.db")
    return AlarmEngine(db_path=db, cooldown_seconds=60)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_alarm_stored(engine):
    """process(alarm) returns stored=True and alarm appears in DB."""
    alarm = make_alarm()
    result = engine.process(alarm)
    assert result["stored"] is True
    assert result["suppressed"] is False
    assert result["alarm"]["alarm_id"] == alarm["alarm_id"]
    rows = engine.get_alerts()
    assert any(r["alarm_id"] == alarm["alarm_id"] for r in rows)


def test_duplicate_suppressed(engine):
    """Same rule within cooldown window → second call suppressed, not in DB."""
    alarm = make_alarm()
    with patch("alarm_engine_module.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 1, 15, 10, 0, 0)
        engine.process(alarm)
        # 30 seconds later — still within 60s cooldown
        mock_dt.now.return_value = datetime(2024, 1, 15, 10, 0, 30)
        second = make_alarm(alarm_id="ALM_20240115_BBBBBB")
        result = engine.process(second)

    assert result["suppressed"] is True
    assert result["stored"] is False
    rows = engine.get_alerts()
    assert len(rows) == 1


def test_cooldown_expires(engine):
    """After cooldown period, the same rule alarm is stored again."""
    alarm1 = make_alarm()
    alarm2 = make_alarm(alarm_id="ALM_20240115_CCCCCC", timestamp="2024-01-15T10:01:05")

    with patch("alarm_engine_module.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 1, 15, 10, 0, 0)
        engine.process(alarm1)
        # 65 seconds later — cooldown has expired
        mock_dt.now.return_value = datetime(2024, 1, 15, 10, 1, 5)
        result = engine.process(alarm2)

    assert result["stored"] is True
    assert result["suppressed"] is False
    rows = engine.get_alerts()
    assert len(rows) == 2


def test_monitor_alarm_not_suppressed(engine):
    """Monitor alarms (rule_id=None) always bypass cooldown."""
    alarm1 = make_alarm(rule_id=None, source="model_monitor")
    alarm2 = make_alarm(
        alarm_id="ALM_20240115_DDDDDD",
        rule_id=None,
        source="model_monitor",
    )

    with patch("alarm_engine_module.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 1, 15, 10, 0, 0)
        r1 = engine.process(alarm1)
        mock_dt.now.return_value = datetime(2024, 1, 15, 10, 0, 10)
        r2 = engine.process(alarm2)

    assert r1["stored"] is True
    assert r2["stored"] is True
    rows = engine.get_alerts()
    assert len(rows) == 2


def test_filter_by_level(engine):
    """get_alerts(level='CRITICAL') returns only CRITICAL rows."""
    engine.process(make_alarm(alarm_id="ALM_20240115_CRIT01", level="CRITICAL", rule_id="RULE_001"))
    engine.process(make_alarm(alarm_id="ALM_20240115_WARN01", level="WARNING", rule_id="RULE_002"))
    critical = engine.get_alerts(level="CRITICAL")
    assert len(critical) == 1
    assert critical[0]["level"] == "CRITICAL"


def test_filter_by_since(engine):
    """get_alerts(since=...) returns only alarms with timestamp >= since."""
    engine.process(make_alarm(
        alarm_id="ALM_20240115_OLD",
        timestamp="2024-01-15T08:00:00",
        rule_id="RULE_001",
    ))
    engine.process(make_alarm(
        alarm_id="ALM_20240115_NEW",
        timestamp="2024-01-15T12:00:00",
        rule_id="RULE_002",
    ))
    results = engine.get_alerts(since="2024-01-15T10:00:00")
    assert len(results) == 1
    assert results[0]["alarm_id"] == "ALM_20240115_NEW"


def test_summary_counts(engine):
    """Summary by_level matches stored alarms exactly."""
    engine.process(make_alarm(alarm_id="ALM_C1", level="CRITICAL", rule_id="RULE_001"))
    engine.process(make_alarm(alarm_id="ALM_C2", level="CRITICAL", rule_id="RULE_009"))
    engine.process(make_alarm(alarm_id="ALM_C3", level="CRITICAL", rule_id="RULE_003"))
    engine.process(make_alarm(alarm_id="ALM_W1", level="WARNING",  rule_id="RULE_002"))
    engine.process(make_alarm(alarm_id="ALM_W2", level="WARNING",  rule_id="RULE_004"))
    engine.process(make_alarm(alarm_id="ALM_I1", level="INFO",     rule_id="RULE_010"))

    summary = engine.get_summary()
    assert summary["total"] == 6
    assert summary["by_level"]["CRITICAL"] == 3
    assert summary["by_level"]["WARNING"] == 2
    assert summary["by_level"]["INFO"] == 1


def test_sync_marking(engine):
    """mark_synced moves alarm out of get_unsynced() list."""
    alarm = make_alarm()
    engine.process(alarm)
    unsynced_before = engine.get_unsynced()
    assert any(r["alarm_id"] == alarm["alarm_id"] for r in unsynced_before)

    ok = engine.mark_synced(alarm["alarm_id"])
    assert ok is True

    unsynced_after = engine.get_unsynced()
    assert not any(r["alarm_id"] == alarm["alarm_id"] for r in unsynced_after)


def test_pagination(engine):
    """get_alerts with limit/offset returns correct pages."""
    # rule_id=None bypasses cooldown so all 10 alarms are stored
    for i in range(10):
        engine.process(make_alarm(
            alarm_id=f"ALM_2024_PAGE{i:02d}",
            rule_id=None,
            source="model_monitor",
        ))

    page0 = engine.get_alerts(limit=3, offset=0)
    page1 = engine.get_alerts(limit=3, offset=3)
    assert len(page0) == 3
    assert len(page1) == 3
    ids0 = {r["alarm_id"] for r in page0}
    ids1 = {r["alarm_id"] for r in page1}
    assert ids0.isdisjoint(ids1)


def test_unknown_alarm_id(engine):
    """get_alert_by_id returns None for a non-existent alarm_id."""
    result = engine.get_alert_by_id("NONEXISTENT_ID")
    assert result is None
