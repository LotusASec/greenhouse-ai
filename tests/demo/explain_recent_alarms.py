#!/usr/bin/env python3
"""Son alarmlara LLM açıklaması ürettirir (demo aracı).

Central aggregator'dan son CRITICAL/HIGH/WARNING alarmları çeker ve her birini
LLM explainer'a (POST /explain) gönderir. Explainer açıklamayı central.db'ye
kendisi yazar → Grafana "LLM Explanations" paneli ve alarm feed'in
explanation kolonu dolar. LLM_API_KEY yoksa şablon (template) metni üretilir.

Kullanım:
    python3 tests/demo/explain_recent_alarms.py            # 3 seviye × 5 alarm
    python3 tests/demo/explain_recent_alarms.py --limit 10
"""

import argparse

import requests

AGGREGATOR_URL = "http://localhost:9001"
LLM_URL        = "http://localhost:9003"
LEVELS         = ["CRITICAL", "HIGH", "WARNING"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5,
                        help="Seviye başına açıklanacak alarm sayısı")
    args = parser.parse_args()

    total = 0
    for level in LEVELS:
        r = requests.get(f"{AGGREGATOR_URL}/alerts",
                         params={"level": level, "limit": args.limit}, timeout=10)
        r.raise_for_status()
        for alarm in r.json():
            if alarm.get("llm_explanation"):
                continue  # zaten açıklanmış
            payload = {
                "alarm_id":       alarm["alarm_id"],
                "node_id":        alarm["node_id"],
                "timestamp":      alarm["timestamp"],
                "level":          alarm["level"],
                "source":         alarm["source"],
                "rule_id":        alarm.get("rule_id"),
                "trigger_values": alarm.get("trigger_values") or {},
                "synced":         bool(alarm.get("synced", True)),
            }
            resp = requests.post(f"{LLM_URL}/explain", json=payload, timeout=30)
            resp.raise_for_status()
            out = resp.json()
            total += 1
            print(f"{alarm['alarm_id']} [{level:8s}] ({out['source']}): "
                  f"{out['explanation'][:90]}")

    print(f"\n{total} alarm açıklandı — Grafana LLM Explanations panelini yenile.")


if __name__ == "__main__":
    main()
