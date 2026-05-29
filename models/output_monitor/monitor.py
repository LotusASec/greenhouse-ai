"""Model Output Monitor — statistical watchdog.

Sliding window z-score + IQR anomaly detection per PHASE_02C_SPEC.
No persistence: in-memory deque only. Buffer resets on service restart.
"""

from collections import deque
from typing import Optional

import numpy as np


class MonitorService:
    WINDOW_SIZE = 50
    MIN_SAMPLES = 10
    Z_THRESHOLD = 2.5
    ALLOWED_MODELS = frozenset({"disease", "irrigation", "nutrition", "anomaly"})

    def __init__(self) -> None:
        self.buffers: dict[str, deque] = {
            "disease":    deque(maxlen=self.WINDOW_SIZE),
            "irrigation": deque(maxlen=self.WINDOW_SIZE),
            "nutrition":  deque(maxlen=self.WINDOW_SIZE),
            "anomaly":    deque(maxlen=self.WINDOW_SIZE),
        }
        self.event_counts:  dict[str, int] = {k: 0 for k in self.buffers}
        self.anomaly_counts: dict[str, int] = {k: 0 for k in self.buffers}

    def check(
        self,
        model_name: str,
        value: float,
        node_id: str,
        timestamp: str,
    ) -> dict:
        """Evaluate one metric value; return a MonitorEvent dict (MASTER_SPEC §3.6)."""
        buf = self.buffers[model_name]
        buf.append(value)
        self.event_counts[model_name] += 1

        # Warm-up: not enough samples yet
        if len(buf) < self.MIN_SAMPLES:
            return {
                "node_id":     node_id,
                "timestamp":   timestamp,
                "model_name":  model_name,
                "metric_value": value,
                "z_score":     0.0,
                "window_mean": value,
                "window_std":  0.0,
                "is_anomaly":  False,
            }

        arr  = np.array(buf, dtype=float)
        mean = float(np.mean(arr))
        std  = float(np.std(arr))

        # std=0 protection (freeze pattern): add 1e-9 to avoid division by zero
        z = (value - mean) / (std + 1e-9)

        # IQR cross-validation
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        iqr_anomaly = bool((value < q1 - 1.5 * iqr) or (value > q3 + 1.5 * iqr))

        # Both z-score and IQR must agree to reduce false positives
        is_anomaly = bool(abs(z) > self.Z_THRESHOLD and iqr_anomaly)

        if is_anomaly:
            self.anomaly_counts[model_name] += 1

        return {
            "node_id":      node_id,
            "timestamp":    timestamp,
            "model_name":   model_name,
            "metric_value": value,
            "z_score":      round(z, 4),
            "window_mean":  round(mean, 4),
            "window_std":   round(std, 4),
            "is_anomaly":   is_anomaly,
        }

    def get_status(self, model_name: Optional[str] = None) -> dict:
        """Return buffer statistics for one or all models."""
        if model_name is not None:
            return self._model_status(model_name)
        return {k: self._model_status(k) for k in self.buffers}

    def _model_status(self, model_name: str) -> dict:
        buf   = self.buffers[model_name]
        n     = len(buf)
        total = self.event_counts[model_name]

        if n == 0:
            return {
                "buffer_size":  0,
                "window_mean":  0.0,
                "window_std":   0.0,
                "anomaly_rate": 0.0,
                "total_events": total,
                "warming_up":   True,
            }

        arr = np.array(buf, dtype=float)
        return {
            "buffer_size":  n,
            "window_mean":  round(float(np.mean(arr)), 4),
            "window_std":   round(float(np.std(arr)), 4),
            "anomaly_rate": round(self.anomaly_counts[model_name] / total, 4) if total > 0 else 0.0,
            "total_events": total,
            "warming_up":   n < self.MIN_SAMPLES,
        }

    def reset(self, model_name: str) -> None:
        """Clear buffer and counters for a model (used in tests)."""
        self.buffers[model_name].clear()
        self.event_counts[model_name]  = 0
        self.anomaly_counts[model_name] = 0
