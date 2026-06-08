"""Central Aggregator Service — Phase 6.

Pull-based sync: central pulls unsynced alarms from each edge node,
writes them to central SQLite, then marks them synced on the edge.
Never pushes — edge is always the source of truth for raw alarms.
"""

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

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
  synced          INTEGER DEFAULT 1,
  created_at      TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_central_alarms_node_ts ON alarms (node_id, timestamp);

CREATE TABLE IF NOT EXISTS sensor_readings (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  node_id       TEXT    NOT NULL,
  timestamp     TEXT    NOT NULL,
  temperature   REAL,
  humidity      REAL,
  soil_moisture REAL,
  light         REAL,
  ec            REAL,
  ph            REAL,
  created_at    TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_central_sr_node_ts ON sensor_readings (node_id, timestamp);

CREATE TABLE IF NOT EXISTS node_registry (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  node_id       TEXT    UNIQUE NOT NULL,
  gateway_url   TEXT    NOT NULL,
  registered_at TEXT    DEFAULT (datetime('now')),
  last_ping     TEXT,
  last_sync     TEXT,
  status        TEXT    DEFAULT 'unknown'
);
"""


def _get_conn(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema(db_path: str) -> None:
    with _get_conn(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)
    log.info("Central DB schema ready at %s", db_path)


class CentralAggregator:
    def __init__(self, db_path: str, sync_interval: int) -> None:
        self.db_path = db_path
        self.sync_interval = sync_interval
        self.nodes: dict[str, dict] = {}
        self._running = False
        self._total_synced = 0
        self._last_sync_at: Optional[str] = None
        _ensure_schema(db_path)

    # ── Node Management ───────────────────────────────────────────────────────

    def register_node(self, node_id: str, gateway_url: str) -> dict:
        self.nodes[node_id] = {
            "node_id":       node_id,
            "gateway_url":   gateway_url,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "last_ping":     None,
            "last_sync":     None,
            "status":        "registered",
        }
        with _get_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO node_registry
                  (node_id, gateway_url, registered_at, status)
                VALUES (:node_id, :gateway_url, :registered_at, :status)
                """,
                {
                    "node_id":       node_id,
                    "gateway_url":   gateway_url,
                    "registered_at": self.nodes[node_id]["registered_at"],
                    "status":        "registered",
                },
            )
        log.info("Node registered: %s → %s", node_id, gateway_url)
        return self.nodes[node_id]

    def get_all_nodes(self) -> list:
        return list(self.nodes.values())

    def get_node_status(self, node_id: str) -> Optional[dict]:
        return self.nodes.get(node_id)

    # ── Sync ──────────────────────────────────────────────────────────────────

    async def sync_node(self, node_id: str) -> dict:
        """Pull unsynced alarms from edge, write to central DB, mark synced."""
        node = self.nodes.get(node_id)
        if not node:
            return {"node_id": node_id, "synced_count": 0, "error": "not registered"}

        url = node["gateway_url"]
        try:
            async with httpx.AsyncClient() as client:
                # 1. Pull unsynced alarms
                r = await client.get(f"{url}/alerts/unsynced", timeout=5.0)
                r.raise_for_status()
                alarms = r.json()
                if isinstance(alarms, dict):
                    alarms = alarms.get("alarms", [])

                # 2. Write to central DB
                count = self._write_alarms(alarms)

                # 3. Mark synced on edge
                if count > 0:
                    await client.post(f"{url}/alerts/sync-all", timeout=5.0)

                # 4. Update node metadata
                now = datetime.now(timezone.utc).isoformat()
                self.nodes[node_id].update({
                    "last_ping": now,
                    "last_sync": now,
                    "status":    "ok",
                })
                with _get_conn(self.db_path) as conn:
                    conn.execute(
                        "UPDATE node_registry SET last_ping=:t, last_sync=:t, status='ok' WHERE node_id=:n",
                        {"t": now, "n": node_id},
                    )

                self._total_synced += count
                self._last_sync_at  = now
                log.info("Synced %d alarms from node %s", count, node_id)
                return {"node_id": node_id, "synced_count": count, "timestamp": now}

        except Exception as exc:
            log.warning("Node %s unreachable: %s", node_id, exc)
            self.nodes[node_id]["status"] = "unreachable"
            with _get_conn(self.db_path) as conn:
                conn.execute(
                    "UPDATE node_registry SET status='unreachable' WHERE node_id=:n",
                    {"n": node_id},
                )
            return {"node_id": node_id, "synced_count": 0, "error": str(exc)}

    async def sync_all_nodes(self) -> list:
        results = []
        for node_id in list(self.nodes.keys()):
            result = await self.sync_node(node_id)
            results.append(result)
        return results

    async def start_sync_loop(self) -> None:
        self._running = True
        log.info("Sync loop started (interval: %ds, nodes: %s)",
                 self.sync_interval, list(self.nodes.keys()))
        while self._running:
            try:
                await self.sync_all_nodes()
            except Exception as exc:
                log.error("Sync loop error: %s", exc)
            await asyncio.sleep(self.sync_interval)

    def stop_sync_loop(self) -> None:
        self._running = False

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_all_alerts(
        self,
        level:   Optional[str] = None,
        source:  Optional[str] = None,
        node_id: Optional[str] = None,
        since:   Optional[str] = None,
        limit:   int = 50,
        offset:  int = 0,
    ) -> list:
        with _get_conn(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM alarms
                WHERE (:level   IS NULL OR level   = :level)
                  AND (:source  IS NULL OR source  = :source)
                  AND (:node_id IS NULL OR node_id = :node_id)
                  AND (:since   IS NULL OR timestamp >= :since)
                ORDER BY timestamp DESC
                LIMIT :limit OFFSET :offset
                """,
                {
                    "level":   level,
                    "source":  source,
                    "node_id": node_id,
                    "since":   since,
                    "limit":   min(int(limit), 200),
                    "offset":  int(offset),
                },
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("trigger_values"):
                try:
                    d["trigger_values"] = json.loads(d["trigger_values"])
                except Exception:
                    pass
            d["synced"] = bool(d.get("synced", 1))
            result.append(d)
        return result

    def get_alerts_summary(self) -> dict:
        with _get_conn(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM alarms").fetchone()["n"]
            by_node = {
                r["node_id"]: r["n"]
                for r in conn.execute(
                    "SELECT node_id, COUNT(*) AS n FROM alarms GROUP BY node_id"
                ).fetchall()
            }
            by_level = {
                r["level"]: r["n"]
                for r in conn.execute(
                    "SELECT level, COUNT(*) AS n FROM alarms GROUP BY level"
                ).fetchall()
            }
        return {
            "total":    total,
            "by_node":  by_node,
            "by_level": by_level,
            "sync_running":    self._running,
            "total_synced":    self._total_synced,
            "last_sync_at":    self._last_sync_at,
        }

    def get_status(self) -> dict:
        return {
            "sync_running":    self._running,
            "node_count":      len(self.nodes),
            "nodes":           list(self.nodes.keys()),
            "total_synced":    self._total_synced,
            "last_sync_at":    self._last_sync_at,
            "sync_interval_s": self.sync_interval,
        }

    def update_alarm_explanation(self, alarm_id: str, explanation: str) -> bool:
        with _get_conn(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE alarms SET llm_explanation=:e WHERE alarm_id=:id",
                {"e": explanation, "id": alarm_id},
            )
        return cur.rowcount > 0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _write_alarms(self, alarms: list) -> int:
        count = 0
        with _get_conn(self.db_path) as conn:
            for alarm in alarms:
                tv = alarm.get("trigger_values", {})
                if isinstance(tv, dict):
                    tv = json.dumps(tv)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO alarms
                      (alarm_id, node_id, timestamp, level, source,
                       rule_id, trigger_values, llm_explanation, synced)
                    VALUES
                      (:alarm_id, :node_id, :timestamp, :level, :source,
                       :rule_id, :trigger_values, :llm_explanation, 1)
                    """,
                    {
                        "alarm_id":       alarm.get("alarm_id"),
                        "node_id":        alarm.get("node_id"),
                        "timestamp":      alarm.get("timestamp"),
                        "level":          alarm.get("level"),
                        "source":         alarm.get("source"),
                        "rule_id":        alarm.get("rule_id"),
                        "trigger_values": tv,
                        "llm_explanation": alarm.get("llm_explanation"),
                    },
                )
                count += 1
        return count
