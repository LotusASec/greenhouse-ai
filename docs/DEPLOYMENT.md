# Deployment Guide — Greenhouse AI

---

## Prerequisites

| Dependency | Minimum Version | Check |
|-----------|----------------|-------|
| Docker | 24.0 | `docker --version` |
| Docker Compose | v2.20 | `docker compose version` |
| Python | 3.11 (local dev only) | `python3 --version` |
| curl | any | `curl --version` |
| Disk space | 10 GB | |
| RAM | 8 GB | |

---

## Installation

```bash
# 1. Clone
git clone <repo-url>
cd greenhouse-ai

# 2. Configure
cp .env.example .env
# Edit .env — mandatory: set LLM_API_KEY if you want real LLM explanations
# Leave LLM_API_KEY empty to use template fallback (works without API key)

# 3. Start
./scripts/start_demo.sh
```

The startup script:
1. Runs `docker compose up -d` (all services)
2. Polls /health on all services (max 10 minutes)
3. Seeds LSTM historical data
4. Prints all service URLs

---

## Configuration (.env variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | _(empty)_ | OpenAI-compatible API key. Leave empty → template fallback |
| `LLM_MODEL` | `gpt-4o-mini` | Model ID for LLM explainer |
| `LLM_MAX_TOKENS` | `200` | Max tokens per explanation |
| `CENTRAL_SYNC_INTERVAL` | `30` | Seconds between central sync pulls from edge |
| `SYNC_INTERVAL_SECONDS` | `30` | Seconds between edge → central alarm push |
| `ALARM_COOLDOWN_SECONDS` | `60` | Cooldown before same rule can fire again (per node) |
| `LOG_LEVEL` | `INFO` | `DEBUG | INFO | WARNING | ERROR` |
| `DATASET_PATH` | _(empty)_ | Path to PlantVillage dataset root (enables image sim) |

---

## Service Ports

| Service | Port |
|---------|------|
| Edge 1 Gateway | 8100 |
| Edge 1 Disease | 8101 |
| Edge 1 Irrigation | 8102 |
| Edge 1 Nutrition | 8103 |
| Edge 1 Anomaly | 8104 |
| Edge 1 Monitor | 8105 |
| Edge 1 Fusion | 8106 |
| Edge 1 Alarm | 8107 |
| Edge 2 Gateway | 8200 |
| Central Gateway | 9000 |
| Central Aggregator | 9001 |
| Central LSTM | 9002 |
| Central LLM | 9003 |
| Central Grafana | 3000 |
| Edge 1 Grafana | 3001 |
| Edge 2 Grafana | 3002 |

---

## Starting Specific Services

```bash
# Start only edge node 1 (without central)
docker compose up edge1_gateway edge1_disease edge1_irrigation \
                   edge1_nutrition edge1_anomaly edge1_output_monitor \
                   edge1_fusion edge1_alarm edge1_grafana

# Start only central layer
docker compose up central_gateway central_aggregator central_lstm central_llm grafana

# Start a single service for debugging
docker compose up edge1_disease
```

---

## Health Check

```bash
# Quick table of all service statuses
./scripts/health_check.sh

# Single service
curl http://localhost:8100/health
```

---

## Grafana Credentials

| Field | Value |
|-------|-------|
| URL | http://localhost:3000 (central), :3001 (edge 1), :3002 (edge 2) |
| Username | `admin` |
| Password | `greenhouse` |
| Anonymous | Viewer access enabled (no login required) |

---

## Common Issues

### Services not starting

```bash
docker compose logs <service-name>
```

Most common causes:
- Training services (`edge1_irrigation_train` etc.) haven't finished yet → wait a few minutes
- Port already in use → check with `ss -tlnp | grep <port>` and stop conflicting process

### Grafana shows no data

1. Check Grafana containers are running: `docker ps --filter name=grafana`
2. Run `./scripts/health_check.sh` — verify edge1_gateway (8100) is healthy
3. The continuous simulation loop writes to the DB — give it 30 seconds

### Central dashboard has no alarms

Central receives alarms via sync. The aggregator pulls from edge every 30 seconds.
Wait 30-60 seconds after startup for first sync.

### LLM returns template text

The `LLM_API_KEY` in `.env` is empty or invalid. The template fallback produces
Turkish-language explanations using rule-based logic. This is fully functional for demo.
Set a valid `LLM_API_KEY` for real GPT explanations.

### Edge autonomous mode not working

The edge gateway's continuous loop requires the SensorSimulator. Check:
```bash
docker compose logs edge1_gateway | grep -i "simulator\|loop"
```

### Clean restart

```bash
# Remove all containers AND volumes (deletes all DB data)
docker compose down -v

# Rebuild and restart
./scripts/start_demo.sh
```

---

## Volumes

| Volume | Container path | Contents |
|--------|---------------|----------|
| `edge1_db` | `/data/greenhouse.db` | Node 1 sensor readings, alarms, monitor events |
| `edge2_db` | `/data/greenhouse.db` | Node 2 data |
| `central_db` | `/data/central.db` | Aggregated alarms, node registry |
| `edge1_irrigation_weights` | `/app/model_weights` | Trained XGBoost model |
| `edge1_nutrition_weights` | `/app/model_weights` | Trained Random Forest model |
| `edge1_anomaly_weights` | `/app/model_weights` | Trained Isolation Forest model |

Volumes persist across `docker compose down` (without `-v`).
Use `docker compose down -v` to start completely fresh.
