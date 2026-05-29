"""Edge Gateway service — HTTP layer only.

Orchestrates the full inference cycle and exposes the edge API.
MASTER_SPEC.md Section 7 — endpoints frozen.
"""

import logging
import os
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

SERVICE_NAME = "edge_gateway"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8100"))
NODE_ID = os.getenv("NODE_ID")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

if not NODE_ID:
    raise EnvironmentError("NODE_ID environment variable is not set")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)


# --- Schemas ---
class SensorValues(BaseModel):
    temperature: float
    humidity: float
    soil_moisture: float
    light: float
    ec: float
    ph: float


class PredictRequest(BaseModel):
    node_id: str
    timestamp: str
    sensors: SensorValues
    image_base64: str = ""


@app.on_event("startup")
async def on_startup() -> None:
    log.info("Starting %s (node: %s) on port %d", SERVICE_NAME, NODE_ID, SERVICE_PORT)


@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@app.post("/predict")
async def predict(payload: PredictRequest) -> Dict[str, Any]:
    # TODO: Phase 6 — call disease, irrigation, nutrition, anomaly, monitor, fusion, alarm
    return {"status": "stub", "node_id": NODE_ID}


@app.get("/threshold")
async def get_threshold() -> Dict[str, Any]:
    # TODO: Phase 6 — return current fusion rule thresholds from rules.yaml
    return {"status": "stub", "node_id": NODE_ID}


@app.get("/logs")
async def get_logs() -> Dict[str, Any]:
    # TODO: Phase 6 — query sensor_readings and model_outputs from SQLite
    return {"status": "stub", "node_id": NODE_ID, "items": []}


@app.get("/alerts")
async def get_alerts() -> Dict[str, Any]:
    # TODO: Phase 6 — query alarms table from SQLite
    return {"status": "stub", "node_id": NODE_ID, "items": []}


@app.get("/sync")
async def sync() -> List[Dict[str, Any]]:
    # TODO: Phase 6 — return unsynced alarms for central to pull
    return []
