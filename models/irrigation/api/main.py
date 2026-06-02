"""
Irrigation model service — HTTP layer only.

Business logic lives in inference.py (IrrigationInference).
MASTER_SPEC.md Section 3.3 — schema frozen.
"""
import sys
import os

# Docker ve local çalışma dizini uyuşmazlığını çözen akıllı yol tanımlaması
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import IrrigationInference

SERVICE_NAME    = "edge_irrigation"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT    = int(os.getenv("SERVICE_PORT", "8102"))
MODEL_DIR       = os.getenv("MODEL_DIR", "model_weights")
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)


# ---------------------------------------------------------------------------
# Lifespan: model loaded once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s on port %d — loading model from %s", SERVICE_NAME, SERVICE_PORT, MODEL_DIR)
    try:
        app.state.inference = IrrigationInference(model_dir=MODEL_DIR)
        log.info("Model loaded: %s from %s", app.state.inference.model_name, MODEL_DIR)
    except FileNotFoundError as exc:
        log.error("Model load failed: %s", exc)
        app.state.inference = None
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
# Input / output schemas — MASTER_SPEC §3.3 (frozen)
# ---------------------------------------------------------------------------

class SensorValues(BaseModel):
    temperature:   float
    humidity:      float
    soil_moisture: float
    light:         float
    ec:            float
    ph:            float


class IrrigationInput(BaseModel):
    node_id:   str
    timestamp: str
    sensors:   SensorValues


class IrrigationOutput(BaseModel):
    model:             str
    node_id:           str
    timestamp:         str
    irrigate:          bool
    amount_liters:     float
    confidence:        float
    inference_time_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    model_ok = (
        app.state.inference is not None and app.state.inference.loaded
    )
    return {
        "status":  "ok" if model_ok else "error",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
    }


@app.post("/predict", response_model=IrrigationOutput)
async def predict(payload: IrrigationInput) -> IrrigationOutput:
    if app.state.inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = app.state.inference.predict(payload.model_dump())
    return IrrigationOutput(**result)
