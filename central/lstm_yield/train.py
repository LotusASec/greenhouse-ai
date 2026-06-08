#!/usr/bin/env python3
"""LSTM Yield Model Training — Phase 6.

Reads sensor_readings from central SQLite, builds 7-day windows,
trains a 2-layer LSTM, saves model weights.

Usage:
    python train.py --db-path /data/central.db
    python train.py  (reads CENTRAL_DB_PATH from env)
"""

import argparse
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("lstm-train")

FEATURE_COLS = ["temperature", "humidity", "soil_moisture", "light", "ec", "ph"]
SEQ_LEN      = 7      # days
MIN_DAYS     = 14
MODEL_DIR    = Path(__file__).parent / "model_weights"
MODEL_DIR.mkdir(exist_ok=True)


class LSTMYieldModel(nn.Module):
    def __init__(self, input_size: int = 6, hidden_size: int = 64,
                 num_layers: int = 2, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.2,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _load_sensor_data(db_path: str, node_id: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT date(timestamp) AS day,
               AVG(temperature)   AS temperature,
               AVG(humidity)      AS humidity,
               AVG(soil_moisture) AS soil_moisture,
               AVG(light)         AS light,
               AVG(ec)            AS ec,
               AVG(ph)            AS ph
        FROM sensor_readings
        WHERE node_id = ?
        GROUP BY date(timestamp)
        ORDER BY day ASC
        """,
        (node_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _build_windows(daily_data: list[dict]) -> tuple:
    """Build (X, y) for training. y = synthetic yield score from sensor averages."""
    X, y = [], []
    features = np.array([[r[c] for c in FEATURE_COLS] for r in daily_data], dtype=np.float32)

    # Normalize
    mins  = features.min(axis=0)
    maxs  = features.max(axis=0)
    denom = np.where(maxs - mins > 0, maxs - mins, 1.0)
    norm  = (features - mins) / denom

    for i in range(len(norm) - SEQ_LEN):
        window = norm[i : i + SEQ_LEN]
        X.append(window)
        # Synthetic yield label: higher is better conditions
        temp_score = 1.0 - abs(features[i + SEQ_LEN - 1][0] - 24.0) / 10.0
        hum_score  = 1.0 - abs(features[i + SEQ_LEN - 1][1] - 65.0) / 25.0
        ec_score   = 1.0 - abs(features[i + SEQ_LEN - 1][4] - 2.2)  / 1.5
        score      = float(np.clip((temp_score + hum_score + ec_score) / 3.0, 0, 1))
        y.append([score])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train(db_path: str, epochs: int = 20) -> dict:
    node_ids = ["greenhouse_01", "greenhouse_02"]
    all_X, all_y = [], []

    for node_id in node_ids:
        daily = _load_sensor_data(db_path, node_id)
        if len(daily) < MIN_DAYS:
            log.warning("Node %s has only %d days (need %d)", node_id, len(daily), MIN_DAYS)
            continue
        X, y = _build_windows(daily)
        all_X.append(X)
        all_y.append(y)

    if not all_X:
        raise ValueError(f"No node has enough data (need ≥{MIN_DAYS} days each)")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    log.info("Training on %d windows from %d nodes", len(X_all), len(all_X))

    X_t = torch.tensor(X_all)
    y_t = torch.tensor(y_all)

    model     = LSTMYieldModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            log.info("Epoch %d/%d  loss=%.4f", epoch, epochs, loss.item())

    save_path = MODEL_DIR / "yield_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_cols":     FEATURE_COLS,
        "seq_len":          SEQ_LEN,
        "trained_at":       datetime.now(timezone.utc).isoformat(),
        "final_loss":       loss.item(),
    }, save_path)
    log.info("Model saved to %s (loss=%.4f)", save_path, loss.item())
    return {"status": "trained", "path": str(save_path), "final_loss": loss.item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=os.getenv("CENTRAL_DB_PATH", "/data/central.db"))
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    result = train(args.db_path, args.epochs)
    print(result)
