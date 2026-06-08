"""Central Gateway — HTTP layer only.

Orchestrates all central services on startup.
Proxies client requests to aggregator (:9001), lstm (:9002), llm (:9003).
MASTER_SPEC.md Section 7 — endpoints frozen.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

SERVICE_NAME    = "central_gateway"
SERVICE_VERSION = "0.6.0"
SERVICE_PORT    = int(os.getenv("SERVICE_PORT", "9000"))
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")

AGG_URL  = os.getenv("AGGREGATOR_SERVICE_URL", "http://central_aggregator:9001")
LSTM_URL = os.getenv("LSTM_SERVICE_URL",       "http://central_lstm:9002")
LLM_URL  = os.getenv("LLM_SERVICE_URL",        "http://central_llm:9003")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s on port %d", SERVICE_NAME, SERVICE_PORT)
    yield
    log.info("%s shutting down", SERVICE_NAME)


app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AlarmObject(BaseModel):
    alarm_id:        str
    node_id:         str
    timestamp:       str
    level:           str
    source:          str
    rule_id:         Optional[str] = None
    trigger_values:  Dict[str, Any] = {}
    llm_explanation: Optional[str] = None
    synced:          bool = False


class NodeRegisterRequest(BaseModel):
    node_id:     str
    gateway_url: str


class ThresholdUpdate(BaseModel):
    field: str
    value: float


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _proxy_get(url: str, params: dict = None) -> Any:
    # Filter None values — httpx sends them as the string "None" otherwise
    clean_params = {k: v for k, v in (params or {}).items() if v is not None} or None
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=clean_params, timeout=8.0)
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")


async def _proxy_post(url: str, body: dict = None) -> Any:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=body or {}, timeout=15.0)
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")


async def _check_service(name: str, url: str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{url}/health", timeout=2.0)
            return "ok" if r.status_code == 200 else "error"
    except Exception:
        return "unreachable"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    statuses = dict(zip(
        ["aggregator", "lstm", "llm"],
        await asyncio.gather(
            _check_service("aggregator", AGG_URL),
            _check_service("lstm",       LSTM_URL),
            _check_service("llm",        LLM_URL),
        ),
    ))
    all_ok = all(s == "ok" for s in statuses.values())
    return {
        "status":   "ok" if all_ok else "degraded",
        "service":  SERVICE_NAME,
        "version":  SERVICE_VERSION,
        "services": statuses,
    }


@app.get("/status")
async def status():
    return await _proxy_get(f"{AGG_URL}/status")


# ── Nodes ─────────────────────────────────────────────────────────────────────

@app.get("/nodes")
async def get_nodes():
    return await _proxy_get(f"{AGG_URL}/nodes")


@app.get("/nodes/{node_id}/status")
async def get_node_status(node_id: str):
    return await _proxy_get(f"{AGG_URL}/nodes/{node_id}/status")


@app.get("/nodes/{node_id}/alerts")
async def get_node_alerts(
    node_id: str,
    level:  Optional[str] = Query(default=None),
    limit:  int = Query(default=20),
    offset: int = Query(default=0),
):
    return await _proxy_get(
        f"{AGG_URL}/nodes/{node_id}/alerts",
        params={"level": level, "limit": limit, "offset": offset},
    )


@app.post("/nodes/register")
async def register_node(body: NodeRegisterRequest):
    return await _proxy_post(f"{AGG_URL}/nodes/register", body.model_dump())


# ── Alerts ────────────────────────────────────────────────────────────────────

@app.get("/alerts")
async def get_alerts(
    level:   Optional[str] = Query(default=None),
    source:  Optional[str] = Query(default=None),
    node_id: Optional[str] = Query(default=None),
    since:   Optional[str] = Query(default=None),
    limit:   int = Query(default=20, ge=1, le=200),
    offset:  int = Query(default=0, ge=0),
):
    return await _proxy_get(
        f"{AGG_URL}/alerts",
        params={
            "level": level, "source": source, "node_id": node_id,
            "since": since, "limit": limit, "offset": offset,
        },
    )


@app.get("/alerts/summary")
async def get_alerts_summary():
    return await _proxy_get(f"{AGG_URL}/alerts/summary")


# ── Yield ─────────────────────────────────────────────────────────────────────

@app.get("/yield")
async def yield_all():
    return await _proxy_get(f"{LSTM_URL}/yield")


@app.get("/yield/{node_id}")
async def yield_for_node(node_id: str):
    return await _proxy_get(f"{LSTM_URL}/yield/{node_id}")


# ── LLM ───────────────────────────────────────────────────────────────────────

@app.post("/explain")
async def explain(payload: AlarmObject):
    return await _proxy_post(f"{LLM_URL}/explain", payload.model_dump())
