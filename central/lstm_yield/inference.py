"""LSTM Yield Inference — Phase 6."""

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

FEATURE_COLS = ["temperature", "humidity", "soil_moisture", "light", "ec", "ph"]
SEQ_LEN      = 7
MIN_DAYS     = 14
MODEL_PATH   = Path(__file__).parent / "model_weights" / "yield_model.pth"


def _load_model():
    if not MODEL_PATH.exists():
        return None, None
    from train import LSTMYieldModel
    ckpt  = torch.load(MODEL_PATH, map_location="cpu")
    model = LSTMYieldModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _count_days(db_path: str, node_id: str) -> int:
    try:
        conn = sqlite3.connect(db_path)
        row  = conn.execute(
            "SELECT COUNT(DISTINCT date(timestamp)) AS d FROM sensor_readings WHERE node_id=?",
            (node_id,),
        ).fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return 0


def _get_daily_avg(db_path: str, node_id: str, days: int = SEQ_LEN) -> Optional[np.ndarray]:
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            """
            SELECT AVG(temperature) AS temperature,
                   AVG(humidity)    AS humidity,
                   AVG(soil_moisture) AS soil_moisture,
                   AVG(light)       AS light,
                   AVG(ec)          AS ec,
                   AVG(ph)          AS ph
            FROM (
                SELECT date(timestamp) AS day, temperature, humidity,
                       soil_moisture, light, ec, ph
                FROM sensor_readings
                WHERE node_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            )
            GROUP BY day
            ORDER BY day DESC
            LIMIT ?
            """,
            (node_id, days * 24, days),
        ).fetchall()
        conn.close()
        if len(rows) < days:
            return None
        data = np.array([[r[i] for i in range(6)] for r in rows[::-1]], dtype=np.float32)
        return data
    except Exception as exc:
        log.error("_get_daily_avg failed: %s", exc)
        return None


def predict(node_id: str, db_path: str) -> dict:
    days = _count_days(db_path, node_id)

    if days < MIN_DAYS:
        return {
            "node_id":           node_id,
            "status":            "insufficient_data",
            "min_days_required": MIN_DAYS,
            "days_available":    days,
        }

    model, ckpt = _load_model()
    if model is None:
        return {
            "node_id": node_id,
            "status":  "model_not_trained",
            "message": "Run train.py first to generate model weights",
        }

    data = _get_daily_avg(db_path, node_id)
    if data is None:
        return {
            "node_id": node_id,
            "status":  "insufficient_data",
            "min_days_required": MIN_DAYS,
            "days_available": days,
        }

    # Normalize
    mins  = data.min(axis=0)
    maxs  = data.max(axis=0)
    denom = np.where(maxs - mins > 0, maxs - mins, 1.0)
    norm  = (data - mins) / denom

    x = torch.tensor(norm).unsqueeze(0)          # (1, SEQ_LEN, 6)
    with torch.no_grad():
        score = float(model(x).squeeze())
    score = max(0.0, min(1.0, score))

    confidence = "high" if days > 30 else ("medium" if days > 21 else "low")

    return {
        "node_id":       node_id,
        "status":        "ok",
        "yield_score":   round(score, 4),
        "forecast_date": datetime.now(timezone.utc).date().isoformat(),
        "confidence":    confidence,
        "days_used":     days,
    }
