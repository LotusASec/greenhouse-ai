"""LLM Explainer API — HTTP layer only.

Business logic lives in explainer.py.
Phase 6: POST /explain with template fallback.
"""

import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from explainer import LLMExplainer  # noqa: E402

SERVICE_NAME    = "central_llm"
SERVICE_VERSION = "0.6.0"
SERVICE_PORT    = int(os.getenv("SERVICE_PORT", "9003"))
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")
CENTRAL_DB_PATH = os.getenv("CENTRAL_DB_PATH", "/data/central.db")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s on port %d", SERVICE_NAME, SERVICE_PORT)
    app.state.explainer = LLMExplainer()
    yield


app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AlarmObject(BaseModel):
    alarm_id:       str
    node_id:        str
    timestamp:      str
    level:          str
    source:         str
    rule_id:        Optional[str] = None
    trigger_values: Dict[str, Any] = {}
    llm_explanation: Optional[str] = None
    synced:         bool = False


class ExplainResponse(BaseModel):
    alarm_id:    str
    explanation: str
    source:      str   # "llm" | "template"


@app.get("/health")
async def health():
    exp: LLMExplainer = app.state.explainer
    return {
        "status":        "ok",
        "service":       SERVICE_NAME,
        "version":       SERVICE_VERSION,
        "llm_available": exp.llm_available,
        "llm_model":     exp.model,
    }


@app.post("/explain", response_model=ExplainResponse)
async def explain(payload: AlarmObject) -> ExplainResponse:
    exp: LLMExplainer = app.state.explainer
    result = await exp.explain(payload.model_dump())

    # Persist explanation to central DB
    _update_db_explanation(payload.alarm_id, result["explanation"])

    log.info(
        "Explained alarm %s (source=%s node=%s)",
        payload.alarm_id, result["source"], payload.node_id,
    )
    return ExplainResponse(
        alarm_id=payload.alarm_id,
        explanation=result["explanation"],
        source=result["source"],
    )


def _update_db_explanation(alarm_id: str, explanation: str) -> None:
    try:
        conn = sqlite3.connect(CENTRAL_DB_PATH)
        conn.execute(
            "UPDATE alarms SET llm_explanation=? WHERE alarm_id=?",
            (explanation, alarm_id),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning("Could not update explanation in DB: %s", exc)
