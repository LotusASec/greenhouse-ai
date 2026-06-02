"""
Disease model service — HTTP layer only.

Business logic lives in inference.py (DiseaseInference).
MASTER_SPEC.md Section 3.2 — schema frozen.
Class names come from checkpoint — never hardcoded.
"""
import numpy as np
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import DiseaseInference

SERVICE_NAME    = "edge_disease"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT    = int(os.getenv("SERVICE_PORT", "8101"))
MODEL_PATH      = os.getenv("MODEL_PATH", "model_weights/disease_model.pth")
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s on port %d — loading model from %s", SERVICE_NAME, SERVICE_PORT, MODEL_PATH)
    try:
        app.state.inference = DiseaseInference(model_path=MODEL_PATH)
        log.info("Model loaded: %s | classes: %s",
                 app.state.inference.model_name,
                 app.state.inference.class_names)
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
# Schemas — MASTER_SPEC §3.2 (frozen)
# ---------------------------------------------------------------------------

class DiseaseInput(BaseModel):
    node_id:    str
    timestamp:  str
    image_b64:  str  # base64 encoded image


class ClassProbabilities(BaseModel):
    healthy:      float
    early_blight: float
    late_blight:  float
    leaf_mold:    float
    other:        float


class DiseaseOutput(BaseModel):
    model:              str
    node_id:            str
    timestamp:          str
    top_prediction:     str
    top_confidence:     float
    class_probabilities: ClassProbabilities
    inference_time_ms:  float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    model_ok = app.state.inference is not None and app.state.inference.loaded
    resp = {
        "status":  "ok" if model_ok else "error",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
    }
    if model_ok:
        resp["class_names"] = app.state.inference.class_names
    return resp


@app.post("/predict", response_model=DiseaseOutput)
async def predict(payload: DiseaseInput) -> DiseaseOutput:
    if app.state.inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    result = app.state.inference.predict(payload.model_dump())
    
    # --- ETİKET KAYMASINI ÖNLEYEN AKILLI SÖZLÜK EŞLEME ---
    # Modelden gelen ham çıktıları alıyoruz (örneğin result["class_probabilities"] 
    # içeride {'healthy': 0.14, 'late_blight': 0.85} şeklinde isimli durmalı)
    raw_probs = result["class_probabilities"]
    
    # Eğer inference.py sadece bir liste [0.85, 0.14, ...] dönüyorsa, 
    # onu modelin gerçek class_names listesiyle eşleştiriyoruz:
    if isinstance(raw_probs, (list, np.ndarray)):
        model_classes = app.state.inference.class_names
        raw_probs = {model_classes[i]: float(raw_probs[i]) for i in range(len(model_classes))}
    
    # Pydantic ClassProbabilities şemasına, isimleri tam olarak doğru anahtarlara denk gelecek şekilde paslıyoruz
    # MASTER_SPEC şemasında eksik kalma ihtimaline karşı .get(..., 0.0) güvencesi ekliyoruz
    validated_probabilities = ClassProbabilities(
        healthy      = float(raw_probs.get("healthy", 0.0)),
        early_blight = float(raw_probs.get("early_blight", 0.0)),
        late_blight  = float(raw_probs.get("late_blight", 0.0)),
        leaf_mold    = float(raw_probs.get("leaf_mold", 0.0)),
        other        = float(raw_probs.get("other", 0.0))
    )
    # ---------------------------------------------------------------------------

    return DiseaseOutput(
        model               = result["model"],
        node_id             = result["node_id"],
        timestamp           = result["timestamp"],
        top_prediction      = result["top_prediction"],
        top_confidence      = result["top_confidence"],
        class_probabilities = validated_probabilities,
        inference_time_ms   = result["inference_time_ms"],
    )
