"""LLM Explainer — Phase 6.

Generates Turkish natural-language explanations for alarms.
Tries LLM API first; always falls back to template on failure.
"""

import json
import logging
import os
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Sen bir sera yönetim sisteminin karar destek asistanısın. "
    "Teknik alarm verilerini sera operatörlerine yönelik, anlaşılır Türkçe "
    "açıklamalara çevir. Maksimum 3 cümle kullan. "
    "Her zaman somut aksiyon önerisi ver."
)

USER_PROMPT_TEMPLATE = (
    "Alarm Seviyesi: {level}\n"
    "Alarm Kaynağı: {source}\n"
    "Kural: {rule_id}\n"
    "Tetikleyici Değerler: {trigger_values}\n"
    "Node: {node_id}\n\n"
    "Bu durumu açıkla ve operatöre ne yapması gerektiğini söyle."
)

_TEMPLATES: dict[str, str] = {
    "fungal_risk_critical": (
        "Sera {node_id}'de kritik mantar riski tespit edildi. "
        "Nem oranı eşiğin üzerinde ve yaprak hastalık belirtisi saptandı. "
        "Acilen havalandırmayı artırın ve etkilenen bitkileri izole edin."
    ),
    "irrigation_required": (
        "Sera {node_id}'de toprak nemi kritik seviyeye düştü. "
        "Sulama sistemini devreye alın."
    ),
    "model_output_drift": (
        "Sera {node_id}'de model çıktısında normalden sapma tespit edildi. "
        "Kamera veya sensör donanımını kontrol edin."
    ),
    "ph_nutrition_problem": (
        "Sera {node_id}'de pH ve besin dengesizliği var. "
        "Gübre programını gözden geçirin ve pH'ı ayarlayın."
    ),
    "sensor_anomaly_high_temp": (
        "Sera {node_id}'de sensör anomalisi ve yüksek sıcaklık tespit edildi. "
        "Havalandırma sistemini ve sensör kalibrasyonunu kontrol edin."
    ),
    "ec_nitrogen_critical": (
        "Sera {node_id}'de EC değeri düşük ve azot eksikliği tespit edildi. "
        "Amonyum nitrat gübresi uygulayın, EC değerini 1.8-2.2 arasına çekin."
    ),
    "default": (
        "Sera {node_id}'de {level} seviyesinde alarm üretildi. "
        "Kaynak: {source}. Sistem durumunu kontrol edin ve gereken aksiyonu alın."
    ),
}


class LLMExplainer:
    def __init__(self) -> None:
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.model   = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "200"))
        self.llm_available: Optional[bool] = None  # None = not tested yet

    async def explain(self, alarm: dict) -> dict:
        """Generate explanation. Falls back to template if LLM fails."""
        try:
            if not self.api_key:
                raise ValueError("LLM_API_KEY not set")
            text = await self._call_llm(alarm)
            self.llm_available = True
            source = "llm"
        except Exception as exc:
            log.warning("LLM failed (%s) — using template fallback", exc)
            self.llm_available = False
            text   = self._render_template(alarm)
            source = "template"

        return {
            "alarm_id":    alarm.get("alarm_id"),
            "explanation": text,
            "source":      source,
        }

    async def _call_llm(self, alarm: dict) -> str:
        tv_str = json.dumps(alarm.get("trigger_values", {}), ensure_ascii=False)
        user_msg = USER_PROMPT_TEMPLATE.format(
            level=alarm.get("level", ""),
            source=alarm.get("source", ""),
            rule_id=alarm.get("rule_id") or "N/A",
            trigger_values=tv_str,
            node_id=alarm.get("node_id", ""),
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.3,
        }
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=15.0,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

    def _render_template(self, alarm: dict) -> str:
        rule_id  = alarm.get("rule_id") or ""
        tv       = alarm.get("trigger_values", {})
        node_id  = alarm.get("node_id", "bilinmiyor")
        level    = alarm.get("level", "")
        source   = alarm.get("source", "")

        # Pick the most specific template by rule_id → message_key mapping
        template_key = "default"
        rule_to_key = {
            "RULE_001": "fungal_risk_critical",
            "RULE_002": "irrigation_required",
            "RULE_003": "ph_nutrition_problem",
            "RULE_004": "sensor_anomaly_high_temp",
            "RULE_006": "ec_nitrogen_critical",
        }
        if rule_id in rule_to_key:
            template_key = rule_to_key[rule_id]
        elif source == "model_monitor":
            template_key = "model_output_drift"

        tmpl = _TEMPLATES.get(template_key, _TEMPLATES["default"])
        return tmpl.format(
            node_id=node_id,
            level=level,
            source=source,
            humidity=tv.get("sensor_reading.sensors.humidity", tv.get("humidity", "?")),
            disease=tv.get("top_prediction", "hastalık"),
            soil_moisture=tv.get("sensor_reading.sensors.soil_moisture", "?"),
            amount_liters=tv.get("amount_liters", "?"),
            model_name=tv.get("trigger_model", "model"),
            z_score=float(tv.get("z_score", 0)),
        )
