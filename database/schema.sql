-- Greenhouse AI — SQLite Schema
-- MASTER_SPEC.md Section 4 — run with: sqlite3 <path> < schema.sql
-- Idempotent: CREATE TABLE IF NOT EXISTS on all tables.

-- 4.1 sensor_readings
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

CREATE INDEX IF NOT EXISTS idx_sensor_readings_node_ts
  ON sensor_readings (node_id, timestamp);

-- 4.2 model_outputs
CREATE TABLE IF NOT EXISTS model_outputs (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  node_id            TEXT    NOT NULL,
  timestamp          TEXT    NOT NULL,
  model_name         TEXT    NOT NULL,
  top_prediction     TEXT,
  confidence         REAL,
  class_probabilities TEXT,   -- JSON string
  inference_time_ms  REAL,
  raw_output         TEXT,    -- Full JSON string
  created_at         TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_model_outputs_node_ts
  ON model_outputs (node_id, timestamp);

-- 4.3 alarms
CREATE TABLE IF NOT EXISTS alarms (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  alarm_id        TEXT    UNIQUE NOT NULL,
  node_id         TEXT    NOT NULL,
  timestamp       TEXT    NOT NULL,
  level           TEXT    NOT NULL,
  source          TEXT    NOT NULL,
  rule_id         TEXT,
  trigger_values  TEXT,   -- JSON string
  llm_explanation TEXT,
  synced          INTEGER DEFAULT 0,  -- 0: false, 1: true
  created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_alarms_node_ts
  ON alarms (node_id, timestamp);

-- 4.4 monitor_events
CREATE TABLE IF NOT EXISTS monitor_events (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  node_id      TEXT    NOT NULL,
  timestamp    TEXT    NOT NULL,
  model_name   TEXT    NOT NULL,
  metric_value REAL,
  z_score      REAL,
  window_mean  REAL,
  window_std   REAL,
  is_anomaly   INTEGER DEFAULT 0,
  created_at   TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_monitor_events_node_ts
  ON monitor_events (node_id, timestamp);

-- 4.5 node_registry (Phase 6 — central layer)
CREATE TABLE IF NOT EXISTS node_registry (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  node_id       TEXT    UNIQUE NOT NULL,
  gateway_url   TEXT    NOT NULL,
  registered_at TEXT    DEFAULT (datetime('now')),
  last_ping     TEXT,
  last_sync     TEXT,
  status        TEXT    DEFAULT 'unknown'
);
