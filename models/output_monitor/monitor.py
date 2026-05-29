"""Model output monitor — stub for Phase 1.

Real monitoring logic implemented in Phase 4 (Z-score + IQR).
"""

from typing import Optional


def check_metric(model_name: str, metric_value: float, node_id: str) -> Optional[dict]:
    # TODO: Phase 4 — sliding window z-score + IQR anomaly detection
    return None
