"""Edge Gateway service — HTTP layer only.

Orchestrates the full edge inference pipeline.
Business logic lives in pipeline.py.
MASTER_SPEC.md Section 7 — endpoints frozen.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from pipeline import EdgePipeline  # noqa: E402

SERVICE_NAME    = "edge_gateway"
SERVICE_VERSION = "0.5.0"
SERVICE_PORT    = int(os.getenv("SERVICE_PORT", "8100"))
NODE_ID         = os.getenv("NODE_ID", "greenhouse_01")
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s (node: %s) on port %d", SERVICE_NAME, NODE_ID, SERVICE_PORT)
    pipeline = EdgePipeline()
    app.state.pipeline = pipeline
    # Start continuous loop as background task
    loop_task = asyncio.create_task(pipeline.start_continuous_loop())
    app.state.loop_task = loop_task
    log.info("Continuous inference loop started")
    yield
    log.info("%s shutting down", SERVICE_NAME)
    pipeline.stop_continuous_loop()
    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class SensorValues(BaseModel):
    temperature:   float
    humidity:      float
    soil_moisture: float
    light:         float
    ec:            float
    ph:            float


class SensorReading(BaseModel):
    node_id:   str
    timestamp: str
    sensors:   SensorValues


class PredictRequest(BaseModel):
    sensor_reading: SensorReading
    image_b64:      Optional[str] = None


class ThresholdUpdate(BaseModel):
    field: str
    value: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    pipeline: EdgePipeline = app.state.pipeline
    service_statuses = await pipeline.check_all_services()
    all_ok = all(s == "ok" for s in service_statuses.values())
    return {
        "status":   "ok" if all_ok else "degraded",
        "service":  SERVICE_NAME,
        "version":  SERVICE_VERSION,
        "node_id":  NODE_ID,
        "services": service_statuses,
    }


@app.post("/predict")
async def predict(payload: PredictRequest) -> Dict[str, Any]:
    pipeline: EdgePipeline = app.state.pipeline
    alarm = await pipeline.run_inference(
        sensor_reading=payload.sensor_reading.model_dump(),
        image_b64=payload.image_b64,
    )
    return alarm


@app.get("/status")
async def status():
    pipeline: EdgePipeline = app.state.pipeline
    service_statuses = await pipeline.check_all_services()
    return {**pipeline.get_status(), "service_statuses": service_statuses}


@app.get("/threshold")
async def get_threshold():
    """Proxy to Fusion Engine GET /rules."""
    import httpx
    fusion_url = os.getenv("FUSION_SERVICE_URL", "http://edge1_fusion:8106")
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{fusion_url}/rules", timeout=5.0)
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Fusion engine unreachable: {exc}")


@app.put("/threshold/{rule_id}")
async def update_threshold(rule_id: str, body: ThresholdUpdate):
    """Proxy to Fusion Engine PUT /rules/{rule_id}/threshold."""
    import httpx
    fusion_url = os.getenv("FUSION_SERVICE_URL", "http://edge1_fusion:8106")
    try:
        async with httpx.AsyncClient() as client:
            r = await client.put(
                f"{fusion_url}/rules/{rule_id}/threshold",
                json={"field": body.field, "value": body.value},
                timeout=5.0,
            )
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Fusion engine unreachable: {exc}")


@app.get("/logs")
async def get_logs(
    limit:  int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """Return recent sensor readings from SQLite."""
    pipeline: EdgePipeline = app.state.pipeline
    return {"items": pipeline.get_logs(limit=limit, offset=offset)}


@app.get("/alerts")
async def get_alerts(
    level:  Optional[str] = Query(default=None),
    source: Optional[str] = Query(default=None),
    since:  Optional[str] = Query(default=None),
    limit:  int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """Proxy to Alarm Engine GET /alerts."""
    import httpx
    alarm_url = os.getenv("ALARM_SERVICE_URL", "http://edge1_alarm:8107")
    params = {"limit": limit, "offset": offset}
    if level:  params["level"]  = level
    if source: params["source"] = source
    if since:  params["since"]  = since
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{alarm_url}/alerts", params=params, timeout=5.0)
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Alarm engine unreachable: {exc}")


@app.get("/sync")
async def sync():
    """Return unsynced alarms — used by central layer in Phase 6."""
    import httpx
    alarm_url = os.getenv("ALARM_SERVICE_URL", "http://edge1_alarm:8107")
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{alarm_url}/alerts/unsynced", timeout=5.0)
            r.raise_for_status()
            data = r.json()
            return data.get("alarms", [])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Alarm engine unreachable: {exc}")


@app.get("/alerts/unsynced")
async def alerts_unsynced():
    """Alias for /sync — central aggregator uses this URL per Phase 6 spec."""
    return await sync()


@app.post("/alerts/sync-all")
async def alerts_sync_all(node_id: Optional[str] = Query(default=None)):
    """Mark all alarms as synced — called by central after pulling."""
    import httpx
    alarm_url = os.getenv("ALARM_SERVICE_URL", "http://edge1_alarm:8107")
    params = {"node_id": node_id} if node_id else {}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{alarm_url}/alerts/sync-all",
                                  params=params, timeout=5.0)
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Alarm engine unreachable: {exc}")
