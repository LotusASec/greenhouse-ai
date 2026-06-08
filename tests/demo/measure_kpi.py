#!/usr/bin/env python3
"""KPI measurement script — Phase 8 T8.4.

Measures all KPIs from MASTER_SPEC §12 that can be automated:
  1. alarm_latency_ms   — POST /predict → alarm response time
  2. llm_latency_ms     — POST /explain → response time
  3. monitor_precision  — inject normal + anomaly values to /monitor
  4. anomaly_precision  — POST normal + anomaly sensors to /predict

Generates KPI_REPORT.md.

Usage:
  python measure_kpi.py
  python measure_kpi.py --edge-url http://localhost:8100
                        --central-url http://localhost:9000
                        --monitor-url http://localhost:8105
                        --anomaly-url http://localhost:8104
"""

import argparse
import asyncio
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

EDGE_URL    = "http://localhost:8100"
CENTRAL_URL = "http://localhost:9000"
MONITOR_URL = "http://localhost:8105"
ANOMALY_URL = "http://localhost:8104"

KPI_TARGETS = {
    "alarm_latency_ms":   2000.0,
    "llm_latency_ms":     5000.0,
    "monitor_precision":  0.85,
    "anomaly_precision":  0.88,
}

NORMAL_SENSORS = {
    "temperature": 24.0, "humidity": 65.0, "soil_moisture": 55.0,
    "light": 700.0, "ec": 2.2, "ph": 6.3,
}
ANOMALY_SENSORS = {
    "temperature": 48.0, "humidity": 99.0, "soil_moisture": 98.0,
    "light": 1950.0, "ec": 4.8, "ph": 1.2,
}
EXPLAIN_ALARM = {
    "alarm_id": "KPI_TEST",
    "node_id": "greenhouse_01",
    "timestamp": "2026-01-01T00:00:00+00:00",
    "level": "HIGH",
    "source": "rule_engine",
    "rule_id": "RULE_001",
    "trigger_values": {"humidity": 88.5},
    "llm_explanation": None,
    "synced": False,
}

ITERATIONS_ALARM  = 10
ITERATIONS_LLM    = 5
MONITOR_NORMAL_N  = 40
MONITOR_ANOMALY_N = 10
ANOMALY_NORMAL_N  = 10
ANOMALY_INJECT_N  = 10


async def measure_alarm_latency(client: httpx.AsyncClient, edge_url: str) -> dict:
    """POST /predict and measure response time × ITERATIONS_ALARM."""
    print(f"\n[1] Alarm latency — {ITERATIONS_ALARM} runs...")
    node_id = "greenhouse_01"
    times: list[float] = []

    for i in range(ITERATIONS_ALARM):
        ts = datetime.now(timezone.utc).isoformat()
        t0 = time.monotonic()
        try:
            r = await client.post(
                f"{edge_url}/predict",
                json={"sensor_reading": {"node_id": node_id, "timestamp": ts, "sensors": NORMAL_SENSORS}},
                timeout=10.0,
            )
            elapsed = (time.monotonic() - t0) * 1000
            if r.status_code == 200:
                times.append(elapsed)
                print(f"  Run {i+1:2d}: {elapsed:.0f}ms")
        except Exception as exc:
            print(f"  Run {i+1:2d}: ERROR — {exc}")

    if not times:
        return {"min": None, "max": None, "avg": None, "p95": None, "passed": False}

    avg = statistics.mean(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    return {
        "min":    min(times),
        "max":    max(times),
        "avg":    avg,
        "p95":    p95,
        "passed": avg <= KPI_TARGETS["alarm_latency_ms"],
    }


async def measure_llm_latency(client: httpx.AsyncClient, central_url: str) -> dict:
    """POST /explain and measure response time × ITERATIONS_LLM."""
    print(f"\n[2] LLM latency — {ITERATIONS_LLM} runs...")
    times: list[float] = []
    template_used = 0

    for i in range(ITERATIONS_LLM):
        t0 = time.monotonic()
        try:
            r = await client.post(
                f"{central_url}/explain",
                json=EXPLAIN_ALARM,
                timeout=30.0,
            )
            elapsed = (time.monotonic() - t0) * 1000
            if r.status_code == 200:
                times.append(elapsed)
                data = r.json()
                src = data.get("source", "unknown")
                if src == "template":
                    template_used += 1
                print(f"  Run {i+1}: {elapsed:.0f}ms  source={src}")
        except Exception as exc:
            print(f"  Run {i+1}: ERROR — {exc}")

    if not times:
        return {"avg": None, "passed": False, "template_fallback": True}

    avg = statistics.mean(times)
    note = f" (template fallback used {template_used}/{len(times)} times)" if template_used else ""
    print(f"  Note: {note.strip()}" if note else "")
    return {
        "avg":               avg,
        "passed":            avg <= KPI_TARGETS["llm_latency_ms"],
        "template_fallback": template_used > 0,
        "template_count":    template_used,
    }


async def measure_monitor_precision(client: httpx.AsyncClient, monitor_url: str) -> dict:
    """
    Inject MONITOR_NORMAL_N normal values then MONITOR_ANOMALY_N spike anomalies.
    Precision = TP / (TP + FP).
    """
    print(f"\n[3] Monitor precision — {MONITOR_NORMAL_N} normal + {MONITOR_ANOMALY_N} anomaly injections...")
    node_id = "greenhouse_01"
    normal_val = 0.88
    anomaly_val = 0.05

    # Reset buffer by sending 50 identical normal values to establish a tight baseline
    print("  Priming buffer...")
    for _ in range(50):
        ts = datetime.now(timezone.utc).isoformat()
        await client.post(
            f"{monitor_url}/monitor/disease",
            json={"node_id": node_id, "timestamp": ts, "metric_value": normal_val},
            timeout=3.0,
        )

    # Send MONITOR_NORMAL_N additional normal values — should be is_anomaly=False
    fn = 0  # false negatives (normal flagged as anomaly)
    tn = 0
    for _ in range(MONITOR_NORMAL_N):
        ts = datetime.now(timezone.utc).isoformat()
        r = await client.post(
            f"{monitor_url}/monitor/disease",
            json={"node_id": node_id, "timestamp": ts, "metric_value": normal_val},
            timeout=3.0,
        )
        if r.status_code == 200:
            ev = r.json()
            if ev.get("is_anomaly"):
                fn += 1
            else:
                tn += 1
    fp = fn  # false positives = normal values incorrectly flagged

    # Send MONITOR_ANOMALY_N spike anomalies — should be is_anomaly=True
    tp = 0
    fn_anomaly = 0
    for _ in range(MONITOR_ANOMALY_N):
        ts = datetime.now(timezone.utc).isoformat()
        r = await client.post(
            f"{monitor_url}/monitor/disease",
            json={"node_id": node_id, "timestamp": ts, "metric_value": anomaly_val},
            timeout=3.0,
        )
        if r.status_code == 200:
            ev = r.json()
            if ev.get("is_anomaly"):
                tp += 1
            else:
                fn_anomaly += 1

    print(f"  Normal → not-anomaly (TN): {tn}")
    print(f"  Normal → anomaly (FP):     {fp}")
    print(f"  Anomaly → anomaly (TP):    {tp}")
    print(f"  Anomaly → not-anomaly (FN): {fn_anomaly}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn_anomaly,
        "precision": precision,
        "passed":    precision >= KPI_TARGETS["monitor_precision"],
    }


async def measure_anomaly_precision(client: httpx.AsyncClient, anomaly_url: str) -> dict:
    """
    POST ANOMALY_NORMAL_N normal + ANOMALY_INJECT_N anomaly sensor readings.
    Precision = TP / (TP + FP).
    """
    print(f"\n[4] Anomaly detection precision — {ANOMALY_NORMAL_N} normal + {ANOMALY_INJECT_N} anomaly...")
    node_id = "greenhouse_01"

    tp = 0; fp = 0; tn = 0; fn = 0

    for _ in range(ANOMALY_NORMAL_N):
        ts = datetime.now(timezone.utc).isoformat()
        r = await client.post(
            f"{anomaly_url}/predict",
            json={"node_id": node_id, "timestamp": ts, "sensors": NORMAL_SENSORS},
            timeout=5.0,
        )
        if r.status_code == 200:
            d = r.json()
            if d.get("is_anomaly"):
                fp += 1
            else:
                tn += 1

    for _ in range(ANOMALY_INJECT_N):
        ts = datetime.now(timezone.utc).isoformat()
        r = await client.post(
            f"{anomaly_url}/predict",
            json={"node_id": node_id, "timestamp": ts, "sensors": ANOMALY_SENSORS},
            timeout=5.0,
        )
        if r.status_code == 200:
            d = r.json()
            if d.get("is_anomaly"):
                tp += 1
            else:
                fn += 1

    print(f"  Normal → not-anomaly (TN): {tn}")
    print(f"  Normal → anomaly (FP):     {fp}")
    print(f"  Anomaly → anomaly (TP):    {tp}")
    print(f"  Anomaly → not-anomaly (FN): {fn}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "passed":    precision >= KPI_TARGETS["anomaly_precision"],
    }


def generate_report(results: dict, output_path: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    alarm = results["alarm"]
    llm   = results["llm"]
    mon   = results["monitor"]
    ano   = results["anomaly"]

    def fmt_ms(v):
        return f"{v:.0f}ms" if v is not None else "N/A"

    def passed_str(v):
        return "✓" if v else "✗"

    lines = [
        "# KPI Ölçüm Raporu",
        f"Tarih: {ts}",
        "",
        "## Özet",
        "",
        "| KPI | Hedef | Ölçülen | Durum |",
        "|-----|-------|---------|-------|",
        f"| Alarm latency (avg) | ≤ {KPI_TARGETS['alarm_latency_ms']:.0f}ms | "
        f"{fmt_ms(alarm.get('avg'))} | {passed_str(alarm.get('passed'))} |",
        f"| LLM latency (avg) | ≤ {KPI_TARGETS['llm_latency_ms']:.0f}ms | "
        f"{fmt_ms(llm.get('avg'))} | {passed_str(llm.get('passed'))} |",
        f"| Monitor precision | ≥ {KPI_TARGETS['monitor_precision']:.2f} | "
        f"{mon.get('precision', 0):.4f} | {passed_str(mon.get('passed'))} |",
        f"| Anomaly precision | ≥ {KPI_TARGETS['anomaly_precision']:.2f} | "
        f"{ano.get('precision', 0):.4f} | {passed_str(ano.get('passed'))} |",
        "",
        "## Alarm Latency Detail",
        f"- Samples: {results.get('alarm_n', 0)}",
        f"- Min: {fmt_ms(alarm.get('min'))}",
        f"- Max: {fmt_ms(alarm.get('max'))}",
        f"- Avg: {fmt_ms(alarm.get('avg'))}",
        f"- P95: {fmt_ms(alarm.get('p95'))}",
        "",
        "## LLM Latency Detail",
        f"- Samples: {ITERATIONS_LLM}",
        f"- Avg: {fmt_ms(llm.get('avg'))}",
        f"- Template fallback: {'Yes' if llm.get('template_fallback') else 'No'}"
        + (f" ({llm.get('template_count', 0)}/{ITERATIONS_LLM} runs)" if llm.get("template_fallback") else ""),
        "",
        "## Monitor Precision Detail",
        f"- Normal samples: {MONITOR_NORMAL_N}",
        f"- Anomaly samples: {MONITOR_ANOMALY_N}",
        f"- TP: {mon.get('tp', 0)}, FP: {mon.get('fp', 0)}, TN: {mon.get('tn', 0)}, FN: {mon.get('fn', 0)}",
        f"- Precision: {mon.get('precision', 0):.4f}",
        "",
        "## Anomaly Precision Detail",
        f"- Normal samples: {ANOMALY_NORMAL_N}",
        f"- Anomaly samples: {ANOMALY_INJECT_N}",
        f"- TP: {ano.get('tp', 0)}, FP: {ano.get('fp', 0)}, TN: {ano.get('tn', 0)}, FN: {ano.get('fn', 0)}",
        f"- Precision: {ano.get('precision', 0):.4f}",
        "",
        "## Non-Automated KPIs",
        "",
        "| KPI | Hedef | Not |",
        "|-----|-------|-----|",
        "| Deployment time | ≤ 10 min | `time docker compose up --build` ile ölç |",
        "| Edge autonomous | ≥ 72h sim | Disconnect test senaryosuna bakın |",
        "| Disease F1 | ≥ 0.85 | Test seti ile `python models/disease/train.py --eval` |",
        "| Irrigation accuracy | ≥ 0.90 | Test seti ile `python models/irrigation/train.py --eval` |",
        "",
    ]

    report = "\n".join(lines)
    Path(output_path).write_text(report, encoding="utf-8")
    print(f"\nKPI_REPORT.md yazıldı: {output_path}")


async def main(edge_url: str, central_url: str, monitor_url: str, anomaly_url: str,
               output: str) -> int:
    print("=" * 50)
    print("Greenhouse AI — KPI Ölçümü")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        alarm_result   = await measure_alarm_latency(client, edge_url)
        llm_result     = await measure_llm_latency(client, central_url)
        monitor_result = await measure_monitor_precision(client, monitor_url)
        anomaly_result = await measure_anomaly_precision(client, anomaly_url)

    results = {
        "alarm":   alarm_result,
        "llm":     llm_result,
        "monitor": monitor_result,
        "anomaly": anomaly_result,
        "alarm_n": ITERATIONS_ALARM,
    }

    print("\n" + "=" * 50)
    print("KPI ÖZET")
    print("=" * 50)
    all_passed = all([
        alarm_result.get("passed"),
        llm_result.get("passed"),
        monitor_result.get("passed"),
        anomaly_result.get("passed"),
    ])

    alarm_avg_str = f"{alarm_result['avg']:.0f}" if alarm_result.get('avg') is not None else "N/A"
    llm_avg_str   = f"{llm_result['avg']:.0f}"   if llm_result.get('avg')   is not None else "N/A"
    print(f"  Alarm latency avg:    {alarm_avg_str}ms"
          f"  {'✓' if alarm_result.get('passed') else '✗'} (≤{KPI_TARGETS['alarm_latency_ms']:.0f}ms)")
    print(f"  LLM latency avg:      {llm_avg_str}ms"
          f"  {'✓' if llm_result.get('passed') else '✗'} (≤{KPI_TARGETS['llm_latency_ms']:.0f}ms)")
    print(f"  Monitor precision:    {monitor_result.get('precision', 0):.4f}"
          f"  {'✓' if monitor_result.get('passed') else '✗'} (≥{KPI_TARGETS['monitor_precision']})")
    print(f"  Anomaly precision:    {anomaly_result.get('precision', 0):.4f}"
          f"  {'✓' if anomaly_result.get('passed') else '✗'} (≥{KPI_TARGETS['anomaly_precision']})")
    print("")
    print(f"  Overall: {'ALL PASS ✓' if all_passed else 'SOME FAIL ✗'}")

    generate_report(results, output)
    return 0 if all_passed else 1


def parse_args():
    parser = argparse.ArgumentParser(description="Greenhouse AI KPI measurement")
    parser.add_argument("--edge-url",    default=EDGE_URL)
    parser.add_argument("--central-url", default=CENTRAL_URL)
    parser.add_argument("--monitor-url", default=MONITOR_URL)
    parser.add_argument("--anomaly-url", default=ANOMALY_URL)
    parser.add_argument("--output",      default="KPI_REPORT.md")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(asyncio.run(main(
        args.edge_url, args.central_url,
        args.monitor_url, args.anomaly_url,
        args.output,
    )))
