"""Edge Gateway Pipeline — Phase 5.

Orchestrates the full inference cycle:
SensorSimulator → [disease, irrigation, nutrition, anomaly] (parallel)
→ OutputMonitor → FusionEngine → AlarmEngine → alarm object.

Never crashes on single model failure — falls back to neutral values.
All service URLs come from environment variables.
"""

import asyncio
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Simulator import via volume-mounted path ──────────────────────────────────
_SIMULATOR_DIR = Path(os.getenv("SIMULATOR_CONFIG_PATH",
                                "/app/simulator/config/node1.yaml")).parent.parent
sys.path.insert(0, str(_SIMULATOR_DIR))

try:
    from sensor_simulator import SensorSimulator
    _SENSOR_SIM_AVAILABLE = True
except ImportError as _e:
    SensorSimulator = None          # type: ignore[misc,assignment]
    _SENSOR_SIM_AVAILABLE = False
    logging.getLogger(__name__).warning("SensorSimulator unavailable: %s", _e)

try:
    from image_simulator import ImageSimulator
    _IMAGE_SIM_AVAILABLE = True
except ImportError as _e:
    ImageSimulator = None           # type: ignore[misc,assignment]
    _IMAGE_SIM_AVAILABLE = False
    logging.getLogger(__name__).warning("ImageSimulator unavailable: %s", _e)

log = logging.getLogger(__name__)

# ── Service URLs — Docker container DNS names as defaults ─────────────────────
DISEASE_SERVICE_URL    = os.getenv("DISEASE_SERVICE_URL",   "http://edge1_disease:8101")
IRRIGATION_SERVICE_URL = os.getenv("IRRIGATION_SERVICE_URL","http://edge1_irrigation:8102")
NUTRITION_SERVICE_URL  = os.getenv("NUTRITION_SERVICE_URL", "http://edge1_nutrition:8103")
ANOMALY_SERVICE_URL    = os.getenv("ANOMALY_SERVICE_URL",   "http://edge1_anomaly:8104")
MONITOR_SERVICE_URL    = os.getenv("MONITOR_SERVICE_URL",   "http://edge1_output_monitor:8105")
FUSION_SERVICE_URL     = os.getenv("FUSION_SERVICE_URL",    "http://edge1_fusion:8106")
ALARM_SERVICE_URL      = os.getenv("ALARM_SERVICE_URL",     "http://edge1_alarm:8107")

# ── Fallback values per TR-32 ─────────────────────────────────────────────────
_FALLBACKS = {
    "disease": {
        "model": "resnet50_disease",
        "top_prediction": "unknown",
        "top_confidence": 0.0,
        "class_probabilities": {
            "healthy": 0.2, "early_blight": 0.2,
            "late_blight": 0.2, "leaf_mold": 0.2, "other": 0.2,
        },
        "inference_time_ms": 0.0,
    },
    "irrigation": {
        "model": "xgboost_irrigation",
        "irrigate": False,
        "amount_liters": 0.0,
        "confidence": 0.0,
        "inference_time_ms": 0.0,
    },
    "nutrition": {
        "model": "rf_nutrition",
        "deficiency_class": "normal",
        "fertilizer_recommendation": "Model unavailable",
        "confidence": 0.0,
        "inference_time_ms": 0.0,
    },
    "anomaly": {
        "model": "isolation_forest_anomaly",
        "is_anomaly": False,
        "anomaly_score": 0.0,
        "inference_time_ms": 0.0,
    },
}


class EdgePipeline:
    """Orchestrates the full edge inference pipeline."""

    def __init__(self) -> None:
        self.disease_url    = DISEASE_SERVICE_URL
        self.irrigation_url = IRRIGATION_SERVICE_URL
        self.nutrition_url  = NUTRITION_SERVICE_URL
        self.anomaly_url    = ANOMALY_SERVICE_URL
        self.monitor_url    = MONITOR_SERVICE_URL
        self.fusion_url     = FUSION_SERVICE_URL
        self.alarm_url      = ALARM_SERVICE_URL
        self.db_path        = os.getenv("DB_PATH", "/data/greenhouse.db")

        # ── Sensor simulator ─────────────────────────────────────────────
        config_path = os.getenv("SIMULATOR_CONFIG_PATH",
                                "/app/simulator/config/node1.yaml")
        if _SENSOR_SIM_AVAILABLE:
            self.simulator = SensorSimulator(config_path, db_path=self.db_path)
            self.emit_interval: int = self.simulator.emit_interval
            log.info("SensorSimulator ready — node: %s, interval: %ds",
                     self.simulator.node_id, self.emit_interval)
        else:
            self.simulator = None
            self.emit_interval = 5
            log.warning("SensorSimulator not available — loop disabled")

        # ── Image simulator (optional — dataset may not exist) ───────────
        dataset_path = os.getenv("DATASET_PATH", "")
        disease_cfg_path = (
            Path(config_path).parent.parent.parent
            / "models" / "disease" / "config" / "column_map.yaml"
        )
        self.image_simulator = None
        if _IMAGE_SIM_AVAILABLE and dataset_path:
            try:
                self.image_simulator = ImageSimulator(
                    config_path,
                    dataset_path,
                    disease_config_path=disease_cfg_path,
                )
                log.info("ImageSimulator ready — dataset: %s", dataset_path)
            except Exception as exc:
                log.warning("ImageSimulator unavailable (%s) — running without images", exc)
        else:
            log.info("DATASET_PATH not set — disease model will use fallback values")

        # ── Loop state ────────────────────────────────────────────────────
        self._running = False
        self._total_inferences = 0
        self._total_alarms = 0
        self._last_inference_at: Optional[str] = None
        self._loop_errors = 0

    # ── Public API ────────────────────────────────────────────────────────────

    async def run_inference(
        self,
        sensor_reading: dict,
        image_b64: Optional[str] = None,
    ) -> dict:
        """Run the full pipeline once. Never raises — returns alarm on any failure."""
        node_id   = sensor_reading["node_id"]
        timestamp = sensor_reading["timestamp"]

        # 1. Parallel model calls (TR-31)
        raw = await asyncio.gather(
            self._call_disease(image_b64, node_id, timestamp),
            self._call_irrigation(sensor_reading),
            self._call_nutrition(sensor_reading),
            self._call_anomaly(sensor_reading),
            return_exceptions=True,
        )

        model_names = ["disease", "irrigation", "nutrition", "anomaly"]
        outputs = list(raw)
        for i, result in enumerate(outputs):
            if isinstance(result, Exception):
                log.error("Model %s failed: %s — using fallback", model_names[i], result)
                outputs[i] = self._get_fallback(model_names[i], node_id, timestamp)

        disease_out, irrigation_out, nutrition_out, anomaly_out = outputs

        # 2. Output Monitor (parallel, best-effort — never blocks pipeline)
        monitor_events = await self._call_monitor_all(
            node_id, timestamp, disease_out, irrigation_out, nutrition_out, anomaly_out
        )

        # 3. Fusion Engine
        fusion_input = self._build_fusion_input(
            sensor_reading, disease_out, irrigation_out,
            nutrition_out, anomaly_out, monitor_events,
        )
        try:
            alarm = await self._call_fusion(fusion_input)
        except Exception as exc:
            log.error("Fusion engine failed: %s — generating INFO fallback alarm", exc)
            alarm = self._fallback_alarm(node_id, timestamp)

        # 4. Alarm Engine
        try:
            result = await self._call_alarm(alarm)
            alarm = result.get("alarm", alarm)
        except Exception as exc:
            log.error("Alarm engine failed: %s — returning alarm without persistence", exc)

        self._total_inferences += 1
        self._total_alarms += 1
        self._last_inference_at = datetime.now(timezone.utc).isoformat()
        return alarm

    async def start_continuous_loop(self) -> None:
        """Continuous inference loop — survives all exceptions per TR-33."""
        if self.simulator is None:
            log.warning("Simulator not available — continuous loop will not run")
            return
        self._running = True
        log.info("Continuous loop started (interval: %ds)", self.emit_interval)
        while self._running:
            try:
                sensor = self.simulator.generate_reading()
                # write to DB (simulator.generate_reading returns but doesn't persist)
                self.simulator._write_db(sensor)

                image_b64: Optional[str] = None
                if self.image_simulator:
                    try:
                        samples = self.image_simulator.sample(n=1)
                        if samples:
                            image_b64 = samples[0]["image_b64"]
                    except Exception as exc:
                        log.warning("Image sample failed: %s", exc)

                await self.run_inference(sensor_reading=sensor, image_b64=image_b64)

            except Exception as exc:
                self._loop_errors += 1
                log.error("Pipeline loop error: %s", exc)

            await asyncio.sleep(self.emit_interval)

    def stop_continuous_loop(self) -> None:
        self._running = False
        log.info("Continuous loop stopping")

    async def check_all_services(self) -> dict:
        """Ping /health on every downstream service."""
        targets = {
            "disease":        self.disease_url,
            "irrigation":     self.irrigation_url,
            "nutrition":      self.nutrition_url,
            "anomaly":        self.anomaly_url,
            "output_monitor": self.monitor_url,
            "fusion":         self.fusion_url,
            "alarm":          self.alarm_url,
        }
        statuses: dict = {}
        async with httpx.AsyncClient() as client:
            tasks = {
                name: client.get(f"{url}/health", timeout=2.0)
                for name, url in targets.items()
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for name, result in zip(tasks.keys(), results):
                if isinstance(result, Exception):
                    statuses[name] = "unreachable"
                else:
                    statuses[name] = "ok" if result.status_code == 200 else "error"
        return statuses

    def get_status(self) -> dict:
        return {
            "loop_running":       self._running,
            "simulator_ready":    self.simulator is not None,
            "image_sim_ready":    self.image_simulator is not None,
            "last_inference_at":  self._last_inference_at,
            "total_inferences":   self._total_inferences,
            "total_alarms":       self._total_alarms,
            "loop_errors":        self._loop_errors,
            "emit_interval_s":    self.emit_interval,
        }

    def get_logs(self, limit: int = 20, offset: int = 0) -> list:
        """Read sensor_readings from SQLite directly."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT node_id, timestamp, temperature, humidity,
                       soil_moisture, light, ec, ph, created_at
                FROM sensor_readings
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
                """,
                {"limit": limit, "offset": offset},
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as exc:
            log.error("get_logs failed: %s", exc)
            return []

    # ── Private model callers ─────────────────────────────────────────────────

    async def _call_disease(
        self, image_b64: Optional[str], node_id: str, timestamp: str
    ) -> dict:
        if not image_b64:
            return self._get_fallback("disease", node_id, timestamp)
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{self.disease_url}/predict",
                json={"node_id": node_id, "timestamp": timestamp, "image_b64": image_b64},
                timeout=5.0,
            )
            r.raise_for_status()
            return r.json()

    async def _call_irrigation(self, sensor_reading: dict) -> dict:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{self.irrigation_url}/predict",
                json=sensor_reading,
                timeout=5.0,
            )
            r.raise_for_status()
            return r.json()

    async def _call_nutrition(self, sensor_reading: dict) -> dict:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{self.nutrition_url}/predict",
                json=sensor_reading,
                timeout=5.0,
            )
            r.raise_for_status()
            return r.json()

    async def _call_anomaly(self, sensor_reading: dict) -> dict:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{self.anomaly_url}/predict",
                json=sensor_reading,
                timeout=5.0,
            )
            r.raise_for_status()
            return r.json()

    async def _call_monitor(
        self, model_name: str, metric_value: float, node_id: str, timestamp: str
    ) -> Optional[dict]:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    f"{self.monitor_url}/monitor/{model_name}",
                    json={"node_id": node_id, "timestamp": timestamp,
                          "metric_value": float(metric_value)},
                    timeout=3.0,
                )
                r.raise_for_status()
                return r.json()
        except Exception as exc:
            log.warning("Monitor call failed for %s: %s", model_name, exc)
            return None

    async def _call_monitor_all(
        self,
        node_id: str,
        timestamp: str,
        disease_out: dict,
        irrigation_out: dict,
        nutrition_out: dict,
        anomaly_out: dict,
    ) -> list:
        metrics = {
            "disease":    disease_out.get("top_confidence", 0.0),
            "irrigation": irrigation_out.get("confidence", 0.0),
            "nutrition":  nutrition_out.get("confidence", 0.0),
            "anomaly":    abs(anomaly_out.get("anomaly_score", 0.0)),
        }
        tasks = [
            self._call_monitor(name, val, node_id, timestamp)
            for name, val in metrics.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, dict)]

    async def _call_fusion(self, fusion_input: dict) -> dict:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{self.fusion_url}/evaluate",
                json=fusion_input,
                timeout=10.0,
            )
            r.raise_for_status()
            return r.json()

    async def _call_alarm(self, alarm: dict) -> dict:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{self.alarm_url}/alarm",
                json=alarm,
                timeout=5.0,
            )
            r.raise_for_status()
            return r.json()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_fallback(self, model_name: str, node_id: str, timestamp: str) -> dict:
        fb = dict(_FALLBACKS[model_name])
        fb["node_id"]  = node_id
        fb["timestamp"] = timestamp
        return fb

    def _build_fusion_input(
        self,
        sensor_reading: dict,
        disease_out: dict,
        irrigation_out: dict,
        nutrition_out: dict,
        anomaly_out: dict,
        monitor_events: list,
    ) -> dict:
        return {
            "node_id":          sensor_reading["node_id"],
            "timestamp":        sensor_reading["timestamp"],
            "sensor_reading":   sensor_reading,
            "disease_output":   disease_out,
            "irrigation_output": irrigation_out,
            "nutrition_output": nutrition_out,
            "anomaly_output":   anomaly_out,
            "monitor_events":   monitor_events,
        }

    def _fallback_alarm(self, node_id: str, timestamp: str) -> dict:
        import uuid
        return {
            "alarm_id":       f"ALM_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6].upper()}",
            "node_id":        node_id,
            "timestamp":      timestamp,
            "level":          "INFO",
            "source":         "rule_engine",
            "rule_id":        None,
            "trigger_values": {},
            "llm_explanation": None,
            "synced":         False,
        }
