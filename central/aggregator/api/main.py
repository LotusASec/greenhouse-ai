"""Central aggregator service — HTTP layer only.

Business logic lives in aggregator.py.
"""

import logging
import os
from typing import Any, Dict

from fastapi import FastAPI

from aggregator import aggregate_nodes

SERVICE_NAME = "central_aggregator"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "9001"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)


@app.on_event("startup")
async def on_startup() -> None:
    log.info("Starting %s on port %d", SERVICE_NAME, SERVICE_PORT)


@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@app.get("/aggregate")
async def aggregate() -> Dict[str, Any]:
    # TODO: Phase 7 — aggregate data from all edge nodes
    return {"status": "stub", "nodes": []}
