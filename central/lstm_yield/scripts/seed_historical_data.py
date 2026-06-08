#!/usr/bin/env python3
"""Seed 21 days of synthetic hourly sensor readings into central DB.

Run before training LSTM:
    python seed_historical_data.py --db-path /data/central.db
"""

import argparse
import math
import os
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sensor_readings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  node_id TEXT NOT NULL, timestamp TEXT NOT NULL,
  temperature REAL, humidity REAL, soil_moisture REAL,
  light REAL, ec REAL, ph REAL,
  created_at TEXT DEFAULT (datetime('now'))
);
"""

NODE_PROFILES = {
    "greenhouse_01": {
        "temperature": (24.0, 0.8), "humidity": (65.0, 2.0),
        "soil_moisture": (55.0, 1.0), "light": (700.0, 10.0),
        "ec": (2.2, 0.05), "ph": (6.3, 0.02),
    },
    "greenhouse_02": {
        "temperature": (23.0, 0.7), "humidity": (58.0, 1.8),
        "soil_moisture": (60.0, 0.9), "light": (680.0, 9.0),
        "ec": (2.1, 0.04), "ph": (6.4, 0.02),
    },
}


def seed(db_path: str, days: int = 21) -> dict:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)

    total = 0
    start = datetime.now(timezone.utc) - timedelta(days=days)
    rng   = random.Random(42)

    for node_id, profile in NODE_PROFILES.items():
        for h in range(days * 24):
            ts    = start + timedelta(hours=h)
            hour  = ts.hour + ts.minute / 60.0
            solar = max(0.0, math.sin((hour - 6.0) * math.pi / 12.0))

            sensors = {}
            for sensor, (mean, std) in profile.items():
                val = rng.gauss(mean, std)
                if sensor == "temperature":
                    val += 3.0 * solar - 1.5
                elif sensor == "light":
                    val = mean * solar + rng.gauss(0, std)
                sensors[sensor] = round(max(0.0, val), 4)

            conn.execute(
                """
                INSERT INTO sensor_readings
                  (node_id, timestamp, temperature, humidity, soil_moisture, light, ec, ph)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    node_id, ts.isoformat(),
                    sensors["temperature"], sensors["humidity"],
                    sensors["soil_moisture"], sensors["light"],
                    sensors["ec"], sensors["ph"],
                ),
            )
            total += 1

    conn.commit()
    conn.close()
    print(f"Seeded {total} readings ({days} days × 24 hours × {len(NODE_PROFILES)} nodes)")
    return {"seeded": total, "days": days, "nodes": list(NODE_PROFILES.keys())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=os.getenv("CENTRAL_DB_PATH", "/data/central.db"))
    parser.add_argument("--days", type=int, default=21)
    args = parser.parse_args()
    seed(args.db_path, args.days)
