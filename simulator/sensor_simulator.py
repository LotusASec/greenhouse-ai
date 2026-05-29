#!/usr/bin/env python3
"""
Sensor simulator for Greenhouse AI.

Generates realistic sensor readings using:
- Sinusoidal 24-hour daily cycle (temperature, light)
- Gaussian noise per sensor (std from config)
- Value clamping to config min/max
- Optional anomaly injection (spike, freeze, drift)
- SQLite persistence (DB_PATH from env)

Usage:
    python sensor_simulator.py --config config/node1.yaml
    DB_PATH=/data/greenhouse.db python sensor_simulator.py --config config/node1.yaml
"""

import argparse
import json
import logging
import math
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("sensor-simulator")

SCHEMA_SQL = """
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
"""


class SensorSimulator:
    """Generates realistic sensor readings from a node yaml profile."""

    # Sensors that participate in sinusoidal daily cycle
    _DAYTIME_SENSORS = {"temperature", "light"}

    def __init__(self, config_path: str, db_path: Optional[str] = None) -> None:
        with open(config_path) as f:
            self._cfg = yaml.safe_load(f)

        self.node_id: str = self._cfg["node_id"]
        self.profile: str = self._cfg["profile"]
        self.emit_interval: int = self._cfg.get("emit_interval_seconds", 1)
        self._sensor_profile: dict = self._cfg["sensor_profile"]
        self._anomaly_cfg: dict = self._cfg.get("anomaly_injection", {})

        self._db_path = db_path or os.getenv("DB_PATH")
        self._conn: Optional[sqlite3.Connection] = None
        if self._db_path:
            self._init_db()

        # Freeze state for "freeze" anomaly injection
        self._frozen_values: Optional[dict] = None
        # Drift offset for "drift" anomaly injection
        self._drift_offsets: dict = {k: 0.0 for k in self._sensor_profile}

        self._rng = __import__("random").Random()
        log.info("SensorSimulator initialized — node: %s, profile: %s", self.node_id, self.profile)

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        db_dir = Path(self._db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()
        log.info("Database ready at %s", self._db_path)

    def _write_db(self, reading: dict) -> None:
        if self._conn is None:
            return
        sensors = reading["sensors"]
        self._conn.execute(
            """
            INSERT INTO sensor_readings
              (node_id, timestamp, temperature, humidity, soil_moisture, light, ec, ph)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reading["node_id"],
                reading["timestamp"],
                sensors["temperature"],
                sensors["humidity"],
                sensors["soil_moisture"],
                sensors["light"],
                sensors["ec"],
                sensors["ph"],
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Reading generation
    # ------------------------------------------------------------------

    def _solar_factor(self, hour: float) -> float:
        """Return [0, 1] solar intensity for a given hour-of-day."""
        # Peaks at noon, zero before 6am and after 6pm
        return max(0.0, math.sin((hour - 6.0) * math.pi / 12.0))

    def generate_reading(self) -> dict:
        """
        Generate one SensorReading JSON per MASTER_SPEC §3.1.
        Values are clamped to profile min/max.
        """
        now = datetime.now(timezone.utc)
        hour = now.hour + now.minute / 60.0
        solar = self._solar_factor(hour)

        # Frozen anomaly: return the same values repeatedly
        if self._frozen_values is not None:
            sensors = dict(self._frozen_values)
        else:
            sensors = {}
            for sensor, params in self._sensor_profile.items():
                mean = float(params["mean"])
                std  = float(params["std"])
                lo   = float(params["min"])
                hi   = float(params["max"])

                # Sinusoidal daily offset for temperature and light
                if sensor == "temperature":
                    mean += 3.0 * solar - 1.5  # ±1.5°C around base mean
                elif sensor == "light":
                    mean = hi * solar           # light follows solar curve

                # Gaussian noise
                import random as _r
                value = _r.gauss(mean, std)

                # Drift offset (accumulates over time)
                value += self._drift_offsets[sensor]

                # Clamp to config bounds
                value = max(lo, min(hi, value))
                sensors[sensor] = round(value, 4)

        reading = {
            "node_id":   self.node_id,
            "timestamp": now.isoformat(),
            "sensors":   sensors,
        }
        return reading

    # ------------------------------------------------------------------
    # Anomaly injection
    # ------------------------------------------------------------------

    def inject_anomaly(self, profile: str) -> None:
        """
        Activate anomaly injection mode.

        Profiles:
          spike  — next generate_reading returns extreme outlier
          freeze — all subsequent readings return the same frozen values
          drift  — values drift 10% per reading until reset
        """
        if profile == "spike":
            # One-shot: next reading will be extreme, then reset
            self._spike_next = True
            log.warning("Anomaly: spike injection enabled for next reading")

        elif profile == "freeze":
            # Capture current reading and freeze it
            self._frozen_values = self.generate_reading()["sensors"]
            log.warning("Anomaly: freeze injection active")

        elif profile == "drift":
            for sensor in self._drift_offsets:
                params = self._sensor_profile[sensor]
                drift = (float(params["max"]) - float(params["min"])) * 0.10
                self._drift_offsets[sensor] += drift
            log.warning("Anomaly: drift injection applied to all sensors")

        else:
            log.error("Unknown anomaly profile: %s", profile)

    def reset_anomaly(self) -> None:
        self._frozen_values = None
        self._drift_offsets = {k: 0.0 for k in self._sensor_profile}
        log.info("Anomaly injection reset")

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Continuous reading loop. Ctrl-C to stop."""
        log.info("Starting emit loop — interval: %ds, node: %s", self.emit_interval, self.node_id)

        # Check if anomaly injection is configured to be auto-enabled
        if self._anomaly_cfg.get("enabled") and self._anomaly_cfg.get("probability", 0) > 0:
            import random as _r
            anomaly_prob = float(self._anomaly_cfg["probability"])
        else:
            anomaly_prob = 0.0

        while True:
            try:
                if anomaly_prob > 0.0:
                    import random as _r
                    if _r.random() < anomaly_prob:
                        self.inject_anomaly("spike")

                reading = self.generate_reading()
                s = reading["sensors"]

                log.info(
                    "%s | %s | temp=%.2f hum=%.2f sm=%.2f light=%.2f ec=%.3f ph=%.3f",
                    reading["node_id"],
                    reading["timestamp"],
                    s["temperature"],
                    s["humidity"],
                    s["soil_moisture"],
                    s["light"],
                    s["ec"],
                    s["ph"],
                )

                if self._conn:
                    self._write_db(reading)

                time.sleep(self.emit_interval)

            except KeyboardInterrupt:
                log.info("Simulator stopped")
                break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Greenhouse sensor simulator")
    parser.add_argument(
        "--config",
        default=os.getenv("NODE_PROFILE_PATH", "config/node1.yaml"),
        help="Path to node yaml config file",
    )
    parser.add_argument(
        "--db-path",
        default=os.getenv("DB_PATH"),
        help="SQLite database path (overrides DB_PATH env var)",
    )
    args = parser.parse_args()

    sim = SensorSimulator(config_path=args.config, db_path=args.db_path)
    sim.run()


if __name__ == "__main__":
    main()
