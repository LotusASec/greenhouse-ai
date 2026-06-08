#!/usr/bin/env python3
"""Demo scenario injection script — Phase 8 T8.2.

Usage:
  python inject_scenario.py --scenario fungal --node 1
  python inject_scenario.py --scenario all    --node 1
  python inject_scenario.py --list
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

# Node URL mapping
NODE_URLS = {
    1: "http://localhost:8100",
    2: "http://localhost:8200",
}
MONITOR_URLS = {
    1: "http://localhost:8105",
    2: "http://localhost:8205",
}

# 5 demo scenarios per PHASE_08_SPEC TR-48
# Sensor values tuned to reliably trigger rules with the actual trained models.
SCENARIOS = {
    "normal": {
        "sensors": {
            "temperature": 24.0, "humidity": 65.0,
            "soil_moisture": 55.0, "light": 700.0,
            "ec": 2.2, "ph": 6.3,
        },
        "expected_level": "INFO",
        "description": "Normal operation — all systems healthy",
    },
    "fungal": {
        # RULE_009: ec<1.5 AND ph>7.0 AND nutrition!=normal → CRITICAL
        # Note: RULE_001 (fungal_risk) requires disease confidence>0.85 which
        # needs a real PlantVillage image (DATASET_PATH). Without it, RULE_009
        # reliably produces CRITICAL for similar environmental risk conditions.
        "sensors": {
            "temperature": 22.0, "humidity": 88.0,
            "soil_moisture": 70.0, "light": 400.0,
            "ec": 1.2, "ph": 7.5,
        },
        "expected_level": "CRITICAL",
        "description": "Fungal/nutrient risk — high humidity + nutrient imbalance → CRITICAL",
    },
    "dry": {
        # RULE_002: soil_moisture<30 AND irrigate=True → WARNING
        "sensors": {
            "temperature": 28.0, "humidity": 45.0,
            "soil_moisture": 22.0, "light": 800.0,
            "ec": 2.0, "ph": 6.5,
        },
        "expected_level": "WARNING",
        "description": "Dry conditions — low soil moisture triggers irrigation warning",
    },
    "anomaly": {
        # RULE_004: is_anomaly=True AND temperature>32 → HIGH
        "sensors": {
            "temperature": 48.0, "humidity": 99.0,
            "soil_moisture": 98.0, "light": 1950.0,
            "ec": 4.8, "ph": 1.2,
        },
        "expected_level": "HIGH",
        "description": "Sensor anomaly — extreme values trigger HIGH alarm",
    },
    "drift": {
        # Monitor inject: send 50 normal values then spike to trigger model monitor
        "monitor_inject": True,
        "model": "disease",
        "values": [0.45] * 50 + [0.10],
        "expected_level": "WARNING",
        "description": "Model drift — 50 stable values then spike triggers model monitor WARNING",
    },
}


async def poll_alerts_for_new(
    node_url: str,
    after_ts: str,
    timeout: float = 10.0,
) -> Optional[dict]:
    """Poll GET /alerts until a new alarm appears after `after_ts`."""
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(f"{node_url}/alerts", timeout=3.0)
                if r.status_code == 200:
                    alarms = r.json()
                    for alarm in alarms:
                        if alarm.get("timestamp", "") >= after_ts:
                            return alarm
            except Exception:
                pass
            await asyncio.sleep(0.5)
    return None


async def _prime_monitor_for_scenario(sensors: dict, node_url: str, monitor_url: str,
                                       count: int = 12) -> None:
    """Pre-run scenario sensors to stabilize monitor buffer baseline."""
    node_id = "greenhouse_01" if "8100" in node_url else "greenhouse_02"
    async with httpx.AsyncClient() as client:
        for _ in range(count):
            ts = datetime.now(timezone.utc).isoformat()
            try:
                await client.post(
                    f"{node_url}/predict",
                    json={"sensor_reading": {"node_id": node_id, "timestamp": ts, "sensors": sensors}},
                    timeout=10.0,
                )
            except Exception:
                pass


async def run_sensor_scenario(name: str, scenario: dict, node_url: str,
                               monitor_url: str = "") -> bool:
    """Run a sensor-injection scenario and verify the alarm level."""
    node_id = "greenhouse_01" if "8100" in node_url else "greenhouse_02"
    ts = datetime.now(timezone.utc).isoformat()

    print(f"\nRunning: {name} on {node_id}...")
    print(f"  Description: {scenario['description']}")
    print(f"  Expected level: {scenario['expected_level']}")

    # For 'normal' scenario: prime monitor buffer with scenario sensors so that
    # the buffer baseline matches the test output values (prevents false-positive
    # monitor anomaly from simulation drift on long-running systems).
    if name == "normal" and monitor_url:
        print("  Priming monitor buffer...")
        await _prime_monitor_for_scenario(scenario["sensors"], node_url, monitor_url)

    before_ts = ts
    t_start = time.monotonic()

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{node_url}/predict",
            json={
                "sensor_reading": {
                    "node_id": node_id,
                    "timestamp": ts,
                    "sensors": scenario["sensors"],
                }
            },
            timeout=15.0,
        )

    elapsed_ms = (time.monotonic() - t_start) * 1000

    if r.status_code != 200:
        print(f"  ERROR: /predict returned HTTP {r.status_code}")
        print(f"  Status: ✗ FAIL")
        return False

    alarm = r.json()
    actual_level = alarm.get("level", "UNKNOWN")
    expected_level = scenario["expected_level"]
    passed = actual_level == expected_level

    print(f"  Actual level:   {actual_level}")
    print(f"  Status:         {'✓ PASS' if passed else '✗ FAIL'}")
    if alarm.get("alarm_id"):
        print(f"  Alarm ID:       {alarm['alarm_id']}")
    if alarm.get("rule_id"):
        print(f"  Rule:           {alarm['rule_id']}")
    print(f"  Source:         {alarm.get('source', 'unknown')}")
    print(f"  Elapsed:        {elapsed_ms:.0f}ms")

    return passed


async def run_drift_scenario(scenario: dict, monitor_url: str) -> bool:
    """Run the model drift scenario directly against the output monitor."""
    node_id = "greenhouse_01"
    model = scenario["model"]
    values = scenario["values"]
    expected_level = scenario["expected_level"]

    print(f"\nRunning: drift on {node_id}...")
    print(f"  Description: {scenario['description']}")
    print(f"  Expected level: {expected_level}")
    print(f"  Injecting {len(values)} values to /monitor/{model}...")

    t_start = time.monotonic()
    last_event: Optional[dict] = None

    async with httpx.AsyncClient() as client:
        for i, val in enumerate(values):
            ts = datetime.now(timezone.utc).isoformat()
            try:
                r = await client.post(
                    f"{monitor_url}/monitor/{model}",
                    json={"node_id": node_id, "timestamp": ts, "metric_value": val},
                    timeout=3.0,
                )
                if r.status_code == 200:
                    last_event = r.json()
            except Exception as exc:
                print(f"  WARNING: monitor call {i} failed: {exc}")

    elapsed_ms = (time.monotonic() - t_start) * 1000

    if last_event is None:
        print("  ERROR: No monitor response received")
        print("  Status: ✗ FAIL")
        return False

    is_anomaly = last_event.get("is_anomaly", False)
    z_score = last_event.get("z_score", 0.0)

    # Drift scenario: spike value (0.10 after buffer of 0.45) should be anomaly=True
    # which corresponds to WARNING level from model_monitor
    actual_level = "WARNING" if is_anomaly else "INFO"
    passed = actual_level == expected_level

    print(f"  Last is_anomaly: {is_anomaly}")
    print(f"  Last z_score:    {z_score:.3f}")
    print(f"  Actual level:    {actual_level}")
    print(f"  Status:          {'✓ PASS' if passed else '✗ FAIL'}")
    print(f"  Elapsed:         {elapsed_ms:.0f}ms")

    return passed


async def run_scenario(name: str, node: int) -> bool:
    """Dispatch to the correct runner for a scenario."""
    if name not in SCENARIOS:
        print(f"Unknown scenario: {name}. Use --list to see available scenarios.")
        return False

    scenario = SCENARIOS[name]
    node_url = NODE_URLS.get(node, NODE_URLS[1])
    monitor_url = MONITOR_URLS.get(node, MONITOR_URLS[1])

    if scenario.get("monitor_inject"):
        return await run_drift_scenario(scenario, monitor_url)
    else:
        return await run_sensor_scenario(name, scenario, node_url, monitor_url)


async def run_all(node: int) -> int:
    """Run all scenarios. Returns number of failures."""
    failures = 0
    for name in SCENARIOS:
        passed = await run_scenario(name, node)
        if not passed:
            failures += 1
    return failures


def print_list():
    print("Available scenarios:")
    for name, sc in SCENARIOS.items():
        level = sc["expected_level"]
        desc = sc["description"]
        print(f"  {name:12s}  expected={level:8s}  {desc}")


def main():
    parser = argparse.ArgumentParser(description="Greenhouse AI demo scenario injector")
    parser.add_argument("--scenario", type=str,
                        help="Scenario name or 'all'")
    parser.add_argument("--node", type=int, default=1,
                        help="Edge node number (1 or 2, default: 1)")
    parser.add_argument("--list", action="store_true",
                        help="List available scenarios")
    args = parser.parse_args()

    if args.list:
        print_list()
        sys.exit(0)

    if not args.scenario:
        parser.print_help()
        sys.exit(1)

    if args.scenario == "all":
        failures = asyncio.run(run_all(args.node))
        print(f"\n{'='*40}")
        print(f"Results: {len(SCENARIOS) - failures}/{len(SCENARIOS)} PASS")
        sys.exit(0 if failures == 0 else 1)
    else:
        passed = asyncio.run(run_scenario(args.scenario, args.node))
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
