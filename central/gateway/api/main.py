"""Central Gateway service — HTTP layer only.

MASTER_SPEC.md Section 7 — all endpoints frozen.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

SERVICE_NAME = "central_gateway"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "9000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)


class ExplainRequest(BaseModel):
    alarm_id: str
    node_id: str
    timestamp: str
    level: str
    source: str
    rule_id: Optional[str]
    trigger_values: Dict[str, Any]


@app.on_event("startup")
async def on_startup() -> None:
    log.info("Starting %s on port %d", SERVICE_NAME, SERVICE_PORT)


@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@app.get("/nodes")
async def list_nodes() -> List[Dict[str, Any]]:
    # TODO: Phase 7 — return connected edge node list + status
    return []


@app.get("/alerts")
async def all_alerts() -> Dict[str, Any]:
    # TODO: Phase 7 — aggregate alarms from all edge nodes
    return {"status": "stub", "items": []}


@app.get("/yield")
async def yield_forecast() -> Dict[str, Any]:
    # TODO: Phase 7 — proxy to central_lstm /yield
    return {"status": "stub", "forecast": None}


@app.post("/explain")
async def explain(payload: ExplainRequest) -> Dict[str, Any]:
    # TODO: Phase 7 — proxy to central_llm /explain
    return {"status": "stub", "alarm_id": payload.alarm_id, "explanation": None}
