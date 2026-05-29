"""Fusion engine service — HTTP layer only.

Business logic lives in engine.py.
MASTER_SPEC.md Sections 3.7, 3.8 — schemas frozen.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from engine import evaluate

SERVICE_NAME = "edge_fusion"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8106"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)


# --- Input schemas (MASTER_SPEC 3.8 — frozen) ---
class SensorValues(BaseModel):
    temperature: float
    humidity: float
    soil_moisture: float
    light: float
    ec: float
    ph: float


class SensorReading(BaseModel):
    node_id: str
    timestamp: str
    sensors: SensorValues


class DiseaseOutput(BaseModel):
    model: str
    node_id: str
    timestamp: str
    top_prediction: str
    top_confidence: float
    class_probabilities: Dict[str, float]
    inference_time_ms: float


class IrrigationOutput(BaseModel):
    model: str
    node_id: str
    timestamp: str
    irrigate: bool
    amount_liters: float
    confidence: float
    inference_time_ms: float


class NutritionOutput(BaseModel):
    model: str
    node_id: str
    timestamp: str
    deficiency_class: str
    fertilizer_recommendation: str
    confidence: float
    inference_time_ms: float


class AnomalyOutput(BaseModel):
    model: str
    node_id: str
    timestamp: str
    is_anomaly: bool
    anomaly_score: float
    inference_time_ms: float


class MonitorEvent(BaseModel):
    node_id: str
    timestamp: str
    model_name: str
    metric_value: float
    z_score: float
    window_mean: float
    window_std: float
    is_anomaly: bool


class FusionInput(BaseModel):
    node_id: str
    timestamp: str
    sensor_reading: SensorReading
    disease_output: DiseaseOutput
    irrigation_output: IrrigationOutput
    nutrition_output: NutritionOutput
    anomaly_output: AnomalyOutput
    monitor_events: List[MonitorEvent]


# --- Output schema (MASTER_SPEC 3.7 — frozen) ---
class AlarmObject(BaseModel):
    alarm_id: str
    node_id: str
    timestamp: str
    level: str
    source: str
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


@app.post("/evaluate", response_model=Optional[AlarmObject])
async def evaluate_rules(payload: FusionInput):
    result = evaluate(payload.model_dump())
    if result is None:
        return None
    return AlarmObject(**result)
