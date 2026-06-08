# Demo Guide — Greenhouse AI

Step-by-step presentation script for the 5-scene demo.

---

## Prerequisites

System must be running. If not:

```bash
./scripts/start_demo.sh
```

All 15 services healthy — verify with `./scripts/health_check.sh`.

---

## Scene 1 — System Startup

**Talking points:** Single-command deployment; all services healthy in under 10 minutes.

```bash
docker compose down -v
./scripts/start_demo.sh
```

**Show:** `docker compose ps` — all containers `Up (healthy)`.

**Expected output:**
```
All services healthy.
  Central Grafana:  http://localhost:3000
  Edge 1 Grafana:   http://localhost:3001
  Edge API:         http://localhost:8100
  Central API:      http://localhost:9000
```

---

## Scene 2 — Normal Operation

**Talking points:** Sensor data flows automatically; Grafana shows live readings; only INFO alarms.

1. Open http://localhost:3001 (Node 1 Edge Grafana)
2. Show **Sensor Trend** panel — temperature, humidity, soil moisture updating every 5s
3. Show **Alarm History** — mostly INFO level

```bash
python tests/demo/inject_scenario.py --scenario normal --node 1
```

**Expected:**
```
Running: normal on greenhouse_01...
  Expected level: INFO
  Actual level:   INFO
  Status: ✓ PASS
```

---

## Scene 3 — Fungal/Critical Risk

**Talking points:** Multi-condition rule fires; CRITICAL alarm generated; LLM explains in natural language.

```bash
python tests/demo/inject_scenario.py --scenario fungal --node 1
```

**Expected:**
```
Running: fungal on greenhouse_01...
  Expected level: CRITICAL
  Actual level:   CRITICAL
  Status: ✓ PASS
  Rule: RULE_009
```

1. Refresh http://localhost:3001 — Alarm History shows CRITICAL in red
2. Refresh http://localhost:3000 — Central dashboard shows CRITICAL across nodes
3. Show LLM Explanations panel — natural language explanation (if LLM_API_KEY set, otherwise template)

---

## Scene 4 — Model Drift (Silent Anomaly)

**Talking points:** Drift detected without any threshold rule firing; model monitoring adds a second safety layer.

```bash
python tests/demo/inject_scenario.py --scenario drift --node 1
```

**Expected:**
```
Running: drift on greenhouse_01...
  Last is_anomaly: True
  Last z_score:    -7.000
  Actual level:    WARNING
  Status: ✓ PASS
```

**Show:** Grafana Edge → Model Output Monitor panel — z_score anomaly entry.

---

## Scene 5 — Disconnection + Resynchronization

**Talking points:** Edge operates autonomously during outage; alarms buffered locally; zero data loss after reconnect.

```bash
python tests/demo/test_disconnection.py
```

**Expected output:**
```
[1] Verifying system is running...
[2] Stopping central services (simulating outage)...
[3] Injecting 5 alarms on edge (edge autonomous mode)...
[4] Verifying edge autonomous operation...
    Edge autonomous: OK
[5] Restarting central services...
[6] Waiting 35s for sync cycle...
[7] Verifying sync...
RESULT: ✓ PASS — 0 data loss confirmed
```

**Show:** Grafana Central (http://localhost:3000) — alarm count increases after sync.

---

## KPI Summary (optional closing slide)

```bash
python tests/demo/measure_kpi.py
cat KPI_REPORT.md
```

| KPI | Target | Measured |
|-----|--------|----------|
| Alarm latency | ≤ 2000ms | ~100ms |
| LLM latency | ≤ 5000ms | ~10ms (template) |
| Monitor precision | ≥ 0.85 | 1.00 |
| Anomaly precision | ≥ 0.88 | 1.00 |

---

## All Scenarios in One Command

```bash
python tests/demo/inject_scenario.py --scenario all --node 1
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Service not healthy | `docker compose logs <service>` |
| Port already in use | `docker compose down` and retry |
| Grafana shows no data | Verify services are running with `./scripts/health_check.sh` |
| LLM returns template | Set `LLM_API_KEY` in `.env` |
