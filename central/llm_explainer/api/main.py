"""LLM explainer service — HTTP layer only.

Business logic lives in explainer.py.
MASTER_SPEC.md Section 7 (/explain) — endpoint frozen.
"""

import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from explainer import explain_alarm

SERVICE_NAME = "central_llm"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "9003"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)


class ExplainRequest(BaseModel):
    alarm_id: str
    node_id: str
    timestamp: str
    level: str
    source: str
    rule_id: Optional[str]
    trigger_values: Dict[str, Any]


class ExplainResponse(BaseModel):
    alarm_id: str
    explanation: Optional[str]


@app.on_event("startup")
async def on_startup() -> None:
    log.info("Starting %s on port %d", SERVICE_NAME, SERVICE_PORT)


@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@app.post("/explain", response_model=ExplainResponse)
async def explain(payload: ExplainRequest) -> ExplainResponse:
    explanation = explain_alarm(payload.model_dump())
    return ExplainResponse(alarm_id=payload.alarm_id, explanation=explanation)
