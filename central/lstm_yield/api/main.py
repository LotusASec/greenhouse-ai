"""LSTM Yield Service API — HTTP layer only.

Business logic lives in inference.py.
Phase 6: GET /yield/{node_id}, GET /yield
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference import predict as lstm_predict  # noqa: E402

SERVICE_NAME    = "central_lstm"
SERVICE_VERSION = "0.6.0"
SERVICE_PORT    = int(os.getenv("SERVICE_PORT", "9002"))
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")
CENTRAL_DB_PATH = os.getenv("CENTRAL_DB_PATH", "/data/central.db")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)

KNOWN_NODES = ["greenhouse_01", "greenhouse_02"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s on port %d — db: %s", SERVICE_NAME, SERVICE_PORT, CENTRAL_DB_PATH)
    yield


app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    from pathlib import Path
    model_ready = (Path(__file__).parent.parent / "model_weights" / "yield_model.pth").exists()
    return {
        "status":      "ok",
        "service":     SERVICE_NAME,
        "version":     SERVICE_VERSION,
        "model_ready": model_ready,
    }


@app.get("/yield/{node_id}")
async def yield_for_node(node_id: str) -> Dict[str, Any]:
    return lstm_predict(node_id, CENTRAL_DB_PATH)


@app.get("/yield")
async def yield_all() -> Dict[str, Any]:
    results = {nid: lstm_predict(nid, CENTRAL_DB_PATH) for nid in KNOWN_NODES}
    return {"forecasts": results}
