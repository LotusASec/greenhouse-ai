"""Fusion engine service — HTTP layer only.

Business logic lives in engine.py.
MASTER_SPEC.md Sections 3.7, 3.8 — schemas frozen.
"""

import json
import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import FusionEngine

SERVICE_NAME = "edge_fusion"
SERVICE_VERSION = "0.3.0"
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8106"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DB_PATH = os.getenv("DB_PATH", "/data/greenhouse.db")
RULES_PATH = str(Path(__file__).parent.parent / "rules" / "rules.yaml")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s on port %d", SERVICE_NAME, SERVICE_PORT)
    app.state.engine = FusionEngine(RULES_PATH)
    log.info("FusionEngine ready — %d rules loaded", len(app.state.engine.rule_engine.rules))
    _ensure_db()
    yield
    log.info("%s shutting down", SERVICE_NAME)


app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _ensure_db() -> None:
    db_dir = Path(DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alarms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alarm_id TEXT UNIQUE NOT NULL,
            node_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            level TEXT NOT NULL,
            source TEXT NOT NULL,
            rule_id TEXT,
            trigger_values TEXT,
            llm_explanation TEXT,
            synced INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()


def _write_alarm(alarm: dict) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO alarms
              (alarm_id, node_id, timestamp, level, source, rule_id,
               trigger_values, llm_explanation, synced)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                alarm["alarm_id"],
                alarm["node_id"],
                alarm["timestamp"],
                alarm["level"],
                alarm["source"],
                alarm.get("rule_id"),
                json.dumps(alarm.get("trigger_values", {})),
                alarm.get("llm_explanation"),
                1 if alarm.get("synced") else 0,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ── Pydantic schemas (MASTER_SPEC §3.8 — frozen) ─────────────────────────────

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


# ── Output schema (MASTER_SPEC §3.7 — frozen) ────────────────────────────────

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


# ── Threshold update body ─────────────────────────────────────────────────────

class ThresholdUpdate(BaseModel):
    field: str
    value: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    engine: FusionEngine = app.state.engine
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "rules_loaded": len(engine.rule_engine.rules),
    }


@app.post("/evaluate", response_model=AlarmObject)
async def evaluate_rules(payload: FusionInput):
    engine: FusionEngine = app.state.engine
    alarm = engine.process(payload.model_dump())
    _write_alarm(alarm)
    log.info(
        "Alarm generated: %s level=%s rule=%s",
        alarm["alarm_id"], alarm["level"], alarm.get("rule_id"),
    )
    return AlarmObject(**alarm)


@app.get("/rules")
async def get_rules():
    engine: FusionEngine = app.state.engine
    return {"rules": engine.rule_engine.get_rules(), "count": len(engine.rule_engine.rules)}


@app.put("/rules/{rule_id}/threshold")
async def update_threshold(rule_id: str, body: ThresholdUpdate):
    engine: FusionEngine = app.state.engine
    try:
        updated_rule = engine.rule_engine.update_threshold(rule_id, body.field, body.value)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    log.info("Threshold updated: rule=%s field=%s value=%s", rule_id, body.field, body.value)
    return {"status": "updated", "rule": updated_rule}
