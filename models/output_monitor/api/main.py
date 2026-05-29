"""Model Output Monitor service — HTTP layer only.

Business logic lives in monitor.py.
MASTER_SPEC.md Section 3.6 — schema frozen.
"""

import logging
import os

from fastapi import FastAPI
from pydantic import BaseModel

from monitor import check_metric

SERVICE_NAME = "edge_output_monitor"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8105"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)


# --- Input schema ---
class MonitorInput(BaseModel):
    node_id: str
    timestamp: str
    model_name: str
    metric_value: float


# --- Output schema (MASTER_SPEC 3.6 — frozen) ---
class MonitorEvent(BaseModel):
    node_id: str
    timestamp: str
    model_name: str
    metric_value: float
    z_score: float
    window_mean: float
    window_std: float
    is_anomaly: bool


@app.on_event("startup")
async def on_startup() -> None:
    log.info("Starting %s on port %d", SERVICE_NAME, SERVICE_PORT)


@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@app.post("/predict", response_model=MonitorEvent)
async def predict(payload: MonitorInput) -> MonitorEvent:
    result = check_metric(payload.model_name, payload.metric_value, payload.node_id)
    if result is None:
        return MonitorEvent(
            node_id=payload.node_id,
            timestamp=payload.timestamp,
            model_name=payload.model_name,
            metric_value=payload.metric_value,
            z_score=0.0,
            window_mean=0.0,
            window_std=0.0,
            is_anomaly=False,
        )
    return MonitorEvent(**result)
