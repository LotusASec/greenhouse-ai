"""Central Aggregator API — HTTP layer only.

Business logic lives in service.py.
Phase 6: node registration, sync loop, aggregated alerts.
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

from service import CentralAggregator  # noqa: E402

SERVICE_NAME    = "central_aggregator"
SERVICE_VERSION = "0.6.0"
SERVICE_PORT    = int(os.getenv("SERVICE_PORT", "9001"))
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")
CENTRAL_DB_PATH = os.getenv("CENTRAL_DB_PATH", "/data/central.db")
SYNC_INTERVAL   = int(os.getenv("CENTRAL_SYNC_INTERVAL", "30"))

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s on port %d", SERVICE_NAME, SERVICE_PORT)
    agg = CentralAggregator(db_path=CENTRAL_DB_PATH, sync_interval=SYNC_INTERVAL)
    app.state.agg = agg
    # Register nodes from env
    n1 = os.getenv("NODE1_GATEWAY_URL", "http://edge1_gateway:8100")
    n2 = os.getenv("NODE2_GATEWAY_URL", "http://edge2_gateway:8200")
    agg.register_node("greenhouse_01", n1)
    agg.register_node("greenhouse_02", n2)
    # Start sync loop
    task = asyncio.create_task(agg.start_sync_loop())
    app.state.sync_task = task
    log.info("Sync loop started for %d nodes", len(agg.nodes))
    yield
    agg.stop_sync_loop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class NodeRegisterRequest(BaseModel):
    node_id:     str
    gateway_url: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    agg: CentralAggregator = app.state.agg
    return {
        "status":       "ok",
        "service":      SERVICE_NAME,
        "version":      SERVICE_VERSION,
        "node_count":   len(agg.nodes),
        "sync_running": agg._running,
    }


@app.get("/nodes")
async def get_nodes():
    agg: CentralAggregator = app.state.agg
    return {"nodes": agg.get_all_nodes(), "count": len(agg.nodes)}


@app.get("/nodes/{node_id}/status")
async def get_node_status(node_id: str):
    agg: CentralAggregator = app.state.agg
    status = agg.get_node_status(node_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    return status


@app.get("/nodes/{node_id}/alerts")
async def get_node_alerts(
    node_id: str,
    level:  Optional[str] = Query(default=None),
    limit:  int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    agg: CentralAggregator = app.state.agg
    return agg.get_all_alerts(node_id=node_id, level=level, limit=limit, offset=offset)


@app.post("/nodes/register")
async def register_node(body: NodeRegisterRequest):
    agg: CentralAggregator = app.state.agg
    result = agg.register_node(body.node_id, body.gateway_url)
    return {"status": "registered", "node": result}


@app.post("/nodes/{node_id}/sync")
async def sync_node(node_id: str):
    agg: CentralAggregator = app.state.agg
    result = await agg.sync_node(node_id)
    return result


@app.get("/alerts")
async def get_alerts(
    level:   Optional[str] = Query(default=None),
    source:  Optional[str] = Query(default=None),
    node_id: Optional[str] = Query(default=None),
    since:   Optional[str] = Query(default=None),
    limit:   int = Query(default=20, ge=1, le=200),
    offset:  int = Query(default=0, ge=0),
):
    agg: CentralAggregator = app.state.agg
    return agg.get_all_alerts(
        level=level, source=source, node_id=node_id,
        since=since, limit=limit, offset=offset,
    )


@app.get("/alerts/summary")
async def get_alerts_summary():
    agg: CentralAggregator = app.state.agg
    return agg.get_alerts_summary()


@app.get("/status")
async def get_status():
    agg: CentralAggregator = app.state.agg
    return agg.get_status()


@app.post("/alarms/{alarm_id}/explain")
async def update_explanation(alarm_id: str, body: Dict[str, Any]):
    """Called by LLM explainer to write back explanation to central DB."""
    agg: CentralAggregator = app.state.agg
    ok = agg.update_alarm_explanation(alarm_id, body.get("explanation", ""))
    return {"updated": ok, "alarm_id": alarm_id}
