"""Model Output Monitor service — HTTP layer only.

Business logic lives in monitor.py (MonitorService).
MASTER_SPEC.md Section 3.6 — output schema frozen.
"""

import logging
import os
import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from monitor import MonitorService

SERVICE_NAME    = "edge_output_monitor"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT    = int(os.getenv("SERVICE_PORT", "8105"))
DB_PATH         = os.getenv("DB_PATH", "")
NODE_ID         = os.getenv("NODE_ID", "greenhouse_01")
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")

ALLOWED_MODELS = MonitorService.ALLOWED_MODELS

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _init_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute("""
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
        )
    """)
    conn.commit()
    conn.close()


def _write_event(path: str, event: dict) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """INSERT INTO monitor_events
           (node_id, timestamp, model_name, metric_value,
            z_score, window_mean, window_std, is_anomaly)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            event["node_id"],
            event["timestamp"],
            event["model_name"],
            event["metric_value"],
            event["z_score"],
            event["window_mean"],
            event["window_std"],
            1 if event["is_anomaly"] else 0,
        ),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s on port %d", SERVICE_NAME, SERVICE_PORT)
    app.state.monitor = MonitorService()
    if DB_PATH:
        try:
            _init_db(DB_PATH)
            log.info("SQLite initialised at %s", DB_PATH)
        except Exception as exc:
            log.warning("DB init failed (continuing without persistence): %s", exc)
    yield
    log.info("%s shutting down", SERVICE_NAME)


app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Schemas — MASTER_SPEC §3.6 (frozen)
# ---------------------------------------------------------------------------

class MonitorInput(BaseModel):
    node_id:      str
    timestamp:    str
    metric_value: float


class MonitorEvent(BaseModel):
    node_id:      str
    timestamp:    str
    model_name:   str
    metric_value: float
    z_score:      float
    window_mean:  float
    window_std:   float
    is_anomaly:   bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@app.post("/monitor/{model_name}", response_model=MonitorEvent)
async def monitor_model(
    payload: MonitorInput,
    model_name: str = Path(...),
) -> MonitorEvent:
    if model_name not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid model_name '{model_name}'. "
                f"Allowed values: {sorted(ALLOWED_MODELS)}"
            ),
        )

    event = app.state.monitor.check(
        model_name=model_name,
        value=payload.metric_value,
        node_id=payload.node_id,
        timestamp=payload.timestamp,
    )

    if DB_PATH:
        try:
            _write_event(DB_PATH, event)
        except Exception as exc:
            log.warning("DB write failed: %s", exc)

    return MonitorEvent(**event)


@app.get("/status")
async def status_all():
    return app.state.monitor.get_status()


@app.get("/status/{model_name}")
async def status_model(model_name: str = Path(...)):
    if model_name not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid model_name '{model_name}'. "
                f"Allowed values: {sorted(ALLOWED_MODELS)}"
            ),
        )
    return app.state.monitor.get_status(model_name)
