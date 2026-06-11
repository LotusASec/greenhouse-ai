"""Alarm Engine — Phase 4.

Manages alarm persistence, cooldown-based duplicate suppression,
history querying, and sync-status tracking. Makes no alarm decisions
— that is the Fusion Engine's job.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

ALARM_COOLDOWN_SECONDS = int(os.getenv("ALARM_COOLDOWN_SECONDS", "60"))

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS alarms (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  alarm_id        TEXT    UNIQUE NOT NULL,
  node_id         TEXT    NOT NULL,
  timestamp       TEXT    NOT NULL,
  level           TEXT    NOT NULL,
  source          TEXT    NOT NULL,
  rule_id         TEXT,
  trigger_values  TEXT,
  llm_explanation TEXT,
  synced          INTEGER DEFAULT 0,
  created_at      TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_alarms_node_ts ON alarms (node_id, timestamp);
"""


def _get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _get_connection(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    if d.get("trigger_values"):
        try:
            d["trigger_values"] = json.loads(d["trigger_values"])
        except (ValueError, TypeError):
            pass
    d["synced"] = bool(d.get("synced", 0))
    return d


class AlarmEngine:
    def __init__(self, db_path: str, cooldown_seconds: int = ALARM_COOLDOWN_SECONDS) -> None:
        self.db_path = db_path
        self.cooldown = cooldown_seconds
        # key: (node_id, rule_id) → datetime of last stored alarm
        self._cooldown_tracker: dict[tuple, datetime] = {}
        _ensure_schema(db_path)
        log.info("AlarmEngine ready — db=%s cooldown=%ds", db_path, self.cooldown)

    # ── Cooldown ──────────────────────────────────────────────────────────────

    def _is_suppressed(self, alarm: dict) -> bool:
        """Return True if alarm should be suppressed due to cooldown.

        Monitor alarms (rule_id=None) always bypass cooldown.
        """
        rule_id = alarm.get("rule_id")
        if rule_id is None:
            return False

        key = (alarm["node_id"], rule_id)
        last_time = self._cooldown_tracker.get(key)

        if last_time is None:
            self._cooldown_tracker[key] = datetime.now()
            return False

        elapsed = (datetime.now() - last_time).total_seconds()
        if elapsed < self.cooldown:
            log.warning(
                "Suppressed duplicate alarm: node=%s rule=%s elapsed=%.1fs",
                alarm["node_id"], rule_id, elapsed,
            )
            return True

        self._cooldown_tracker[key] = datetime.now()
        return False

    # ── Core process ──────────────────────────────────────────────────────────

    def process(self, alarm: dict) -> dict:
        """Store alarm unless suppressed by cooldown.

        NOTE — intentional dual-write: FusionEngine also persists the same
        alarm to this table at evaluation time (INT-08); this write (INT-10)
        is deduplicated by the alarm_id PRIMARY KEY + INSERT OR IGNORE.
        AlarmEngine owns cooldown/suppression; FusionEngine's write is a
        persistence safety net. Keep both — see INT-08/INT-10 docs.

        Returns:
            {"stored": bool, "suppressed": bool, "alarm": alarm_dict}
        """
        if self._is_suppressed(alarm):
            return {"stored": False, "suppressed": True, "alarm": alarm}

        trigger_json = json.dumps(alarm.get("trigger_values") or {})
        with _get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO alarms
                  (alarm_id, node_id, timestamp, level, source,
                   rule_id, trigger_values, llm_explanation, synced)
                VALUES
                  (:alarm_id, :node_id, :timestamp, :level, :source,
                   :rule_id, :trigger_values, :llm_explanation, :synced)
                """,
                {
                    "alarm_id":       alarm["alarm_id"],
                    "node_id":        alarm["node_id"],
                    "timestamp":      alarm["timestamp"],
                    "level":          alarm["level"],
                    "source":         alarm["source"],
                    "rule_id":        alarm.get("rule_id"),
                    "trigger_values": trigger_json,
                    "llm_explanation": alarm.get("llm_explanation"),
                    "synced":         1 if alarm.get("synced") else 0,
                },
            )

        log.info("Stored alarm %s level=%s", alarm["alarm_id"], alarm["level"])
        return {"stored": True, "suppressed": False, "alarm": alarm}

    # ── Querying ──────────────────────────────────────────────────────────────

    def get_alerts(
        self,
        level: Optional[str] = None,
        source: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        limit = min(int(limit), 200)
        offset = int(offset)
        with _get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM alarms
                WHERE (:level  IS NULL OR level  = :level)
                  AND (:source IS NULL OR source = :source)
                  AND (:since  IS NULL OR timestamp >= :since)
                ORDER BY timestamp DESC
                LIMIT :limit OFFSET :offset
                """,
                {
                    "level":  level,
                    "source": source,
                    "since":  since,
                    "limit":  limit,
                    "offset": offset,
                },
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_alert_by_id(self, alarm_id: str) -> Optional[dict]:
        with _get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM alarms WHERE alarm_id = :alarm_id",
                {"alarm_id": alarm_id},
            ).fetchone()
        return _row_to_dict(row) if row else None

    def get_summary(self) -> dict:
        with _get_connection(self.db_path) as conn:
            total_row = conn.execute("SELECT COUNT(*) AS n FROM alarms").fetchone()
            total = total_row["n"] if total_row else 0

            level_rows = conn.execute(
                "SELECT level, COUNT(*) AS n FROM alarms GROUP BY level"
            ).fetchall()

            source_rows = conn.execute(
                "SELECT source, COUNT(*) AS n FROM alarms GROUP BY source"
            ).fetchall()

            unsynced_row = conn.execute(
                "SELECT COUNT(*) AS n FROM alarms WHERE synced = 0"
            ).fetchone()

            last_row = conn.execute(
                "SELECT timestamp FROM alarms ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()

        by_level = {"CRITICAL": 0, "HIGH": 0, "WARNING": 0, "INFO": 0}
        for r in level_rows:
            by_level[r["level"]] = r["n"]

        by_source: dict = {}
        for r in source_rows:
            by_source[r["source"]] = r["n"]

        return {
            "total": total,
            "by_level": by_level,
            "by_source": by_source,
            "unsynced": unsynced_row["n"] if unsynced_row else 0,
            "last_alarm_at": last_row["timestamp"] if last_row else None,
        }

    # ── Sync ──────────────────────────────────────────────────────────────────

    def mark_synced(self, alarm_id: str) -> bool:
        with _get_connection(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE alarms SET synced = 1 WHERE alarm_id = :alarm_id",
                {"alarm_id": alarm_id},
            )
        return cur.rowcount > 0

    def mark_all_synced(self, node_id: Optional[str] = None) -> int:
        with _get_connection(self.db_path) as conn:
            if node_id:
                cur = conn.execute(
                    "UPDATE alarms SET synced = 1 WHERE synced = 0 AND node_id = :node_id",
                    {"node_id": node_id},
                )
            else:
                cur = conn.execute("UPDATE alarms SET synced = 1 WHERE synced = 0")
        return cur.rowcount

    def get_unsynced(self, node_id: Optional[str] = None) -> list[dict]:
        with _get_connection(self.db_path) as conn:
            if node_id:
                rows = conn.execute(
                    """
                    SELECT * FROM alarms
                    WHERE synced = 0 AND node_id = :node_id
                    ORDER BY timestamp ASC
                    """,
                    {"node_id": node_id},
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM alarms WHERE synced = 0 ORDER BY timestamp ASC"
                ).fetchall()
        return [_row_to_dict(r) for r in rows]
