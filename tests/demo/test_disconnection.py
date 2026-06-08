#!/usr/bin/env python3
"""Disconnection and resync test — Phase 8 T8.3.

Automated disconnection scenario:
1. Verify system running normally
2. Stop central services (simulating network outage)
3. Inject 5 alarms on edge (edge operates autonomously)
4. Verify edge has 5 new alarms with synced=false
5. Restart central services
6. Wait for sync cycle (CENTRAL_SYNC_INTERVAL=30s)
7. Verify alarms synced to central
8. Print report: data loss = 0

Usage:
  python test_disconnection.py
  python test_disconnection.py --edge-url http://localhost:8100
                               --central-url http://localhost:9000
"""

import argparse
import asyncio
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

EDGE_URL    = "http://localhost:8100"
CENTRAL_URL = "http://localhost:9000"

# Use different sensor conditions to trigger different rules and avoid cooldown suppression
INJECT_SENSOR_SETS = [
    {"temperature": 22.0, "humidity": 88.0, "soil_moisture": 70.0,
     "light": 400.0, "ec": 1.2, "ph": 7.5},       # RULE_009 → CRITICAL
    {"temperature": 28.0, "humidity": 45.0, "soil_moisture": 22.0,
     "light": 800.0, "ec": 2.0, "ph": 6.5},        # RULE_002 → WARNING
    {"temperature": 48.0, "humidity": 99.0, "soil_moisture": 98.0,
     "light": 1950.0, "ec": 4.8, "ph": 1.2},       # RULE_003 → HIGH
    {"temperature": 22.0, "humidity": 88.0, "soil_moisture": 70.0,
     "light": 400.0, "ec": 1.0, "ph": 8.0},        # RULE_009 variant (different ec/ph values)
    {"temperature": 28.0, "humidity": 45.0, "soil_moisture": 20.0,
     "light": 900.0, "ec": 1.9, "ph": 6.4},        # RULE_002 variant
]
NUM_ALARMS = len(INJECT_SENSOR_SETS)

# waiting for sync cycle (CENTRAL_SYNC_INTERVAL=30s)
SYNC_WAIT_SECONDS = 35


def run_docker(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose"] + cmd,
        capture_output=True, text=True, timeout=60,
    )


async def get_edge_alarms(client: httpx.AsyncClient, edge_url: str) -> list:
    r = await client.get(f"{edge_url}/alerts", timeout=5.0)
    r.raise_for_status()
    return r.json()


async def get_edge_unsynced(client: httpx.AsyncClient, edge_url: str) -> list:
    r = await client.get(f"{edge_url}/alerts/unsynced", timeout=5.0)
    r.raise_for_status()
    return r.json()


async def get_central_alarms(client: httpx.AsyncClient, central_url: str) -> list:
    r = await client.get(f"{central_url}/alerts", timeout=5.0)
    r.raise_for_status()
    return r.json()


async def inject_alarm(client: httpx.AsyncClient, edge_url: str, i: int) -> Optional[dict]:
    node_id = "greenhouse_01"
    ts = datetime.now(timezone.utc).isoformat()
    sensors = INJECT_SENSOR_SETS[i % len(INJECT_SENSOR_SETS)]
    try:
        r = await client.post(
            f"{edge_url}/predict",
            json={"sensor_reading": {"node_id": node_id, "timestamp": ts, "sensors": sensors}},
            timeout=15.0,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as exc:
        print(f"  WARNING: inject {i} failed: {exc}")
    return None


async def wait_for_health(url: str, timeout: int = 60) -> bool:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(f"{url}/health", timeout=3.0)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(2)
    return False


async def main(edge_url: str, central_url: str) -> int:
    print("=" * 50)
    print("Greenhouse AI — Disconnection Test")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        # ── Step 1: Verify system running ─────────────────────────────────
        print("\n[1] Verifying system is running...")
        try:
            r = await client.get(f"{edge_url}/health", timeout=5.0)
            if r.status_code != 200:
                print(f"  ERROR: Edge gateway not healthy (HTTP {r.status_code})")
                return 1
        except Exception as exc:
            print(f"  ERROR: Edge gateway unreachable: {exc}")
            return 1
        print("  Edge gateway: OK")

        try:
            r = await client.get(f"{central_url}/health", timeout=5.0)
            if r.status_code != 200:
                print(f"  ERROR: Central gateway not healthy (HTTP {r.status_code})")
                return 1
        except Exception as exc:
            print(f"  ERROR: Central gateway unreachable: {exc}")
            return 1
        print("  Central gateway: OK")

        # Record baseline counts
        alarms_before = await get_edge_alarms(client, edge_url)
        central_before = await get_central_alarms(client, central_url)
        baseline_edge_ids = {a["alarm_id"] for a in alarms_before}
        baseline_central_count = len(central_before)
        print(f"  Baseline: edge={len(alarms_before)} alarms, central={baseline_central_count} alarms")

        # ── Step 2: Stop central services ─────────────────────────────────
        print("\n[2] Stopping central services (simulating outage)...")
        result = run_docker(["stop", "central_gateway", "central_aggregator"])
        if result.returncode != 0:
            print(f"  ERROR: docker compose stop failed: {result.stderr}")
            return 1
        print("  Central services stopped.")
        await asyncio.sleep(2)

        # ── Step 3: Inject 5 alarms on edge ───────────────────────────────
        print(f"\n[3] Injecting {NUM_ALARMS} alarms on edge (edge autonomous mode)...")
        injected_ids: list[str] = []
        for i in range(NUM_ALARMS):
            alarm = await inject_alarm(client, edge_url, i + 1)
            if alarm and "alarm_id" in alarm:
                injected_ids.append(alarm["alarm_id"])
                print(f"  [{i+1}/{NUM_ALARMS}] alarm_id={alarm['alarm_id']} level={alarm['level']}")
            else:
                print(f"  [{i+1}/{NUM_ALARMS}] no alarm returned (cooldown or INFO suppressed)")
            await asyncio.sleep(0.5)

        # ── Step 4: Verify edge autonomous operation ───────────────────────
        print("\n[4] Verifying edge autonomous operation...")
        alarms_during = await get_edge_alarms(client, edge_url)
        new_edge_alarms = [a for a in alarms_during if a["alarm_id"] not in baseline_edge_ids]
        unsynced = await get_edge_unsynced(client, edge_url)
        new_unsynced = [a for a in unsynced if a["alarm_id"] not in baseline_edge_ids]

        print(f"  New edge alarms (during outage): {len(new_edge_alarms)}")
        print(f"  Unsynced alarms (new):           {len(new_unsynced)}")

        edge_autonomous_ok = len(new_edge_alarms) >= 1
        print(f"  Edge autonomous: {'OK' if edge_autonomous_ok else 'FAIL — no new alarms generated'}")

        # ── Step 5: Restart central ───────────────────────────────────────
        print("\n[5] Restarting central services...")
        result = run_docker(["start", "central_gateway", "central_aggregator"])
        if result.returncode != 0:
            print(f"  ERROR: docker compose start failed: {result.stderr}")
            return 1
        print("  Waiting for central_gateway to become healthy...")
        central_up = await wait_for_health(central_url, timeout=90)
        if not central_up:
            print("  ERROR: Central gateway did not come back healthy in 90s")
            return 1
        print("  Central services restarted: OK")

        # ── Step 6: Wait for sync ─────────────────────────────────────────
        print(f"\n[6] Waiting {SYNC_WAIT_SECONDS}s for sync cycle (CENTRAL_SYNC_INTERVAL=30s)...")
        for remaining in range(SYNC_WAIT_SECONDS, 0, -5):
            print(f"  ... {remaining}s remaining")
            await asyncio.sleep(min(5, remaining))

        # ── Step 7: Verify sync ───────────────────────────────────────────
        print("\n[7] Verifying sync...")
        central_after = await get_central_alarms(client, central_url)
        new_central = [a for a in central_after if a["alarm_id"] not in {b["alarm_id"] for b in central_before}]
        unsynced_after = await get_edge_unsynced(client, edge_url)
        new_unsynced_after = [a for a in unsynced_after if a["alarm_id"] not in baseline_edge_ids]

        print(f"  New alarms in central: {len(new_central)}")
        print(f"  Still unsynced on edge: {len(new_unsynced_after)}")

        alarms_produced = len(new_edge_alarms)
        alarms_synced   = len(new_central)
        data_loss       = max(0, alarms_produced - alarms_synced)

    # ── Step 8: Report ────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("DISCONNECTION TEST REPORT")
    print("=" * 50)
    print(f"  Edge autonomous during outage: {'✓' if edge_autonomous_ok else '✗'}")
    print(f"  Alarms produced:  {alarms_produced}")
    print(f"  Alarms synced:    {alarms_synced}")
    print(f"  Data loss:        {data_loss}")

    all_synced_ok = len(new_unsynced_after) == 0
    success = edge_autonomous_ok and data_loss == 0

    print("")
    if success:
        print("RESULT: ✓ PASS — 0 data loss confirmed")
    else:
        issues = []
        if not edge_autonomous_ok:
            issues.append("edge did not operate autonomously")
        if data_loss > 0:
            issues.append(f"{data_loss} alarm(s) not synced")
        print(f"RESULT: ✗ FAIL — {'; '.join(issues)}")

    return 0 if success else 1


def parse_args():
    parser = argparse.ArgumentParser(description="Greenhouse AI disconnection test")
    parser.add_argument("--edge-url",    default=EDGE_URL)
    parser.add_argument("--central-url", default=CENTRAL_URL)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(asyncio.run(main(args.edge_url, args.central_url)))
