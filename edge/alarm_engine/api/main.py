"""Alarm engine service — HTTP layer only.

Business logic lives in engine.py.
MASTER_SPEC.md Section 3.7 — alarm schema frozen.
Phase 4: full persistence, cooldown, filtering, sync endpoints.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine import AlarmEngine

SERVICE_NAME = "edge_alarm"
SERVICE_VERSION = "0.4.0"
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8107"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DB_PATH = os.getenv("DB_PATH", "/data/greenhouse.db")
ALARM_COOLDOWN_SECONDS = int(os.getenv("ALARM_COOLDOWN_SECONDS", "60"))

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s on port %d", SERVICE_NAME, SERVICE_PORT)
    app.state.engine = AlarmEngine(
        db_path=DB_PATH,
        cooldown_seconds=ALARM_COOLDOWN_SECONDS,
    )
    log.info(
        "AlarmEngine ready — db=%s cooldown=%ds",
        DB_PATH, ALARM_COOLDOWN_SECONDS,
    )
    yield
    log.info("%s shutting down", SERVICE_NAME)


app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas (MASTER_SPEC §3.7 — frozen) ─────────────────────────────

class AlarmObject(BaseModel):
    alarm_id: str
    node_id: str
    timestamp: str
    level: str        # INFO | WARNING | HIGH | CRITICAL
    source: str       # rule_engine | sensor_anomaly | model_monitor
    rule_id: Optional[str] = None
    trigger_values: Dict[str, Any] = {}
    llm_explanation: Optional[str] = None
    synced: bool = False


class ProcessResult(BaseModel):
    stored: bool
    suppressed: bool
    alarm: AlarmObject


class SyncResult(BaseModel):
    status: str
    alarm_id: str
    updated: bool


class SyncAllResult(BaseModel):
    status: str
    count: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "cooldown_seconds": ALARM_COOLDOWN_SECONDS,
    }


@app.post("/alarm", response_model=ProcessResult)
async def receive_alarm(payload: AlarmObject):
    engine: AlarmEngine = app.state.engine
    result = engine.process(payload.model_dump())
    log.info(
        "Alarm %s — stored=%s suppressed=%s",
        payload.alarm_id, result["stored"], result["suppressed"],
    )
    return ProcessResult(
        stored=result["stored"],
        suppressed=result["suppressed"],
        alarm=AlarmObject(**result["alarm"]),
    )


@app.get("/alerts/summary")
async def get_summary():
    engine: AlarmEngine = app.state.engine
    return engine.get_summary()


@app.get("/alerts/unsynced")
async def get_unsynced(node_id: Optional[str] = Query(default=None)):
    engine: AlarmEngine = app.state.engine
    rows = engine.get_unsynced(node_id=node_id)
    return {"count": len(rows), "alarms": rows}


@app.get("/alerts", response_model=List[AlarmObject])
async def get_alerts(
    level: Optional[str] = Query(default=None),
    source: Optional[str] = Query(default=None),
    since: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    engine: AlarmEngine = app.state.engine
    rows = engine.get_alerts(
        level=level,
        source=source,
        since=since,
        limit=limit,
        offset=offset,
    )
    return [AlarmObject(**r) for r in rows]


@app.get("/alerts/{alarm_id}", response_model=AlarmObject)
async def get_alert(alarm_id: str):
    engine: AlarmEngine = app.state.engine
    row = engine.get_alert_by_id(alarm_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Alarm not found: {alarm_id}")
    return AlarmObject(**row)


@app.post("/alerts/{alarm_id}/sync", response_model=SyncResult)
async def mark_synced(alarm_id: str):
    engine: AlarmEngine = app.state.engine
    updated = engine.mark_synced(alarm_id)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Alarm not found: {alarm_id}")
    log.info("Marked synced: %s", alarm_id)
    return SyncResult(status="synced", alarm_id=alarm_id, updated=True)


@app.post("/alerts/sync-all", response_model=SyncAllResult)
async def sync_all(node_id: Optional[str] = Query(default=None)):
    engine: AlarmEngine = app.state.engine
    count = engine.mark_all_synced(node_id=node_id)
    log.info("Marked all synced: count=%d node_id=%s", count, node_id)
    return SyncAllResult(status="ok", count=count)
