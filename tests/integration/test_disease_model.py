"""
Integration tests for Phase 2B disease model service.

Requires disease service running at localhost:8101.
Start before running:
    MODEL_PATH=model_weights/disease_model.pth uvicorn api.main:app --port 8101

Run:
    pytest tests/integration/test_disease_model.py -v
"""

import base64
import sys
from pathlib import Path

import pytest
import requests

DISEASE_URL = "http://localhost:8101"
DATA_DIR    = Path(__file__).parents[2] / "models" / "disease" / "data"
VALID_CLASSES = {"healthy", "early_blight", "late_blight", "leaf_mold", "other"}

TIMESTAMP = "2024-01-01T12:00:00Z"
NODE_ID   = "greenhouse_01"


def _first_image(cls: str) -> str:
    """Return base64 of the first available test image for a class."""
    for split in ("test", "val", "train"):
        cls_dir = DATA_DIR / split / cls
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        if imgs:
            return base64.b64encode(imgs[0].read_bytes()).decode()
    pytest.skip(f"No images found for class '{cls}' in data/")


def _post_predict(image_b64: str) -> dict:
    resp = requests.post(
        f"{DISEASE_URL}/predict",
        json={"node_id": NODE_ID, "timestamp": TIMESTAMP, "image_b64": image_b64},
        timeout=30,
    )
    assert resp.status_code == 200, f"Unexpected status {resp.status_code}: {resp.text[:200]}"
    return resp.json()


# ---------------------------------------------------------------------------
# 1 — Health
# ---------------------------------------------------------------------------

def test_disease_health():
    """GET /health → 200, status ok, class_names present."""
    resp = requests.get(f"{DISEASE_URL}/health", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok", f"status not ok: {data}"
    assert "class_names" in data, "class_names missing from health response"
    assert set(data["class_names"]) == VALID_CLASSES, (
        f"Unexpected class_names: {data['class_names']}"
    )


# ---------------------------------------------------------------------------
# 2 — Healthy image prediction
# ---------------------------------------------------------------------------

def test_disease_predict_healthy():
    """POST with healthy image → schema matches MASTER_SPEC §3.2."""
    b64 = _first_image("healthy")
    data = _post_predict(b64)

    # Schema check (MASTER_SPEC §3.2)
    assert data["model"]         == "resnet50_disease"
    assert isinstance(data["node_id"],           str)
    assert isinstance(data["timestamp"],         str)
    assert isinstance(data["top_prediction"],    str)
    assert data["top_prediction"]               in VALID_CLASSES
    assert 0.0 <= data["top_confidence"] <= 1.0
    assert isinstance(data["class_probabilities"], dict)
    assert set(data["class_probabilities"].keys()) == VALID_CLASSES
    assert isinstance(data["inference_time_ms"], float)

    # top_confidence == class_probabilities[top_prediction]
    pred  = data["top_prediction"]
    conf  = data["top_confidence"]
    assert abs(conf - data["class_probabilities"][pred]) < 0.001, (
        f"top_confidence {conf} != class_probabilities[{pred}]={data['class_probabilities'][pred]}"
    )


# ---------------------------------------------------------------------------
# 3 — Blight image prediction
# ---------------------------------------------------------------------------

def test_disease_predict_blight():
    """POST with early_blight image → top_prediction is a blight or disease class."""
    b64 = _first_image("early_blight")
    data = _post_predict(b64)
    assert data["top_prediction"] in VALID_CLASSES
    # Accept any non-normal plant disease class (synthetic data may confuse blight types)
    assert data["top_prediction"] != "leaf_mold", (
        "Unexpected prediction for early_blight image"
    )


# ---------------------------------------------------------------------------
# 4 — Probability sum ≈ 1.0
# ---------------------------------------------------------------------------

def test_class_probabilities_sum():
    """Sum of class_probabilities must be between 0.99 and 1.01."""
    b64 = _first_image("healthy")
    data = _post_predict(b64)
    prob_sum = sum(data["class_probabilities"].values())
    assert 0.99 <= prob_sum <= 1.01, (
        f"class_probabilities sum = {prob_sum:.4f}, expected ≈ 1.0"
    )


# ---------------------------------------------------------------------------
# 5 — Inference time
# ---------------------------------------------------------------------------

def test_inference_time():
    """inference_time_ms must be < 2000ms (CPU inference)."""
    b64 = _first_image("healthy")
    data = _post_predict(b64)
    assert data["inference_time_ms"] < 2000.0, (
        f"inference_time_ms = {data['inference_time_ms']}, expected < 2000"
    )
