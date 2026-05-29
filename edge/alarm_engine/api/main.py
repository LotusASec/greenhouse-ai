"""Alarm engine service — HTTP layer only.

Business logic lives in engine.py.
MASTER_SPEC.md Section 3.7 — schema frozen.
"""

import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from engine import store_alarm

SERVICE_NAME = "edge_alarm"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8107"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)


# --- Alarm schema (MASTER_SPEC 3.7 — frozen) ---
class AlarmObject(BaseModel):
    alarm_id: str
    node_id: str
    timestamp: str
    level: str  # INFO | WARNING | HIGH | CRITICAL
    source: str  # rule_engine | sensor_anomaly | model_monitor
    rule_id: Optional[str]
    trigger_values: Dict[str, Any]
    llm_explanation: Optional[str]
    synced: bool


@app.on_event("startup")
async def on_startup() -> None:
    log.info("Starting %s on port %d", SERVICE_NAME, SERVICE_PORT)


@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@app.post("/alarm", response_model=AlarmObject)
async def receive_alarm(payload: AlarmObject) -> AlarmObject:
    result = store_alarm(payload.model_dump())
    if result is None:
        # Placeholder — Phase 5 persists to SQLite
        return payload
    return AlarmObject(**result)
