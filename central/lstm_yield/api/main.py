"""LSTM yield forecast service — HTTP layer only.

Business logic lives in lstm.py.
MASTER_SPEC.md Section 7 (/yield) — endpoint frozen.
"""

import logging
import os
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from lstm import forecast

SERVICE_NAME = "central_lstm"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "9002"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)


class YieldRequest(BaseModel):
    node_ids: List[str]
    time_series: List[Dict[str, Any]]


@app.on_event("startup")
async def on_startup() -> None:
    log.info("Starting %s on port %d", SERVICE_NAME, SERVICE_PORT)


@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@app.post("/yield")
async def yield_forecast(payload: YieldRequest) -> Dict[str, Any]:
    result = forecast(payload.time_series)
    if result is None:
        return {"status": "stub", "forecast": None}
    return result
