# Greenhouse AI

An edge-central hybrid decision support system for smart greenhouse management.
Each greenhouse node runs a set of specialist ML models (disease detection,
irrigation, nutrition, anomaly) locally and remains autonomous during cloud
disconnections. A central layer aggregates multi-node data for yield forecasting
and LLM-powered alarm explanations.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              EDGE LAYER                     │
│         (1 node per greenhouse)             │
│                                             │
│  Simulator → [disease, irrigation,          │
│               nutrition, anomaly]           │
│            → Model Output Monitor           │
│            → Fusion / Rule Engine           │
│            → Alarm Engine                  │
│            → Local Grafana + SQLite         │
│            → Edge Gateway (FastAPI)         │
└──────────────────┬──────────────────────────┘
                   │ HTTP sync
┌──────────────────▼──────────────────────────┐
│             CENTRAL LAYER                   │
│                                             │
│  Aggregator → LSTM Yield → LLM Explainer   │
│  Central Gateway → Central Grafana          │
│  Central SQLite                             │
└─────────────────────────────────────────────┘
```

---

## Directory Structure

```
greenhouse-ai/
├── README.md
├── docker-compose.yml
├── .env.example
├── docs/
│   ├── architecture.md
│   ├── api-reference.md
│   └── tubitak/
├── simulator/
│   ├── sensor_simulator.py
│   ├── image_simulator.py
│   ├── config/
│   │   ├── node1.yaml
│   │   └── node2.yaml
│   └── Dockerfile
├── models/
│   ├── disease/        (ResNet-50, port 8101/8201)
│   ├── irrigation/     (XGBoost,   port 8102/8202)
│   ├── nutrition/      (RandomForest, port 8103/8203)
│   ├── anomaly/        (IsolationForest, port 8104/8204)
│   └── output_monitor/ (Z-score, port 8105/8205)
├── edge/
│   ├── fusion_engine/  (port 8106/8206)
│   ├── alarm_engine/   (port 8107/8207)
│   └── gateway/        (port 8100/8200)
├── central/
│   ├── aggregator/     (port 9001)
│   ├── lstm_yield/     (port 9002)
│   ├── llm_explainer/  (port 9003)
│   └── gateway/        (port 9000)
├── database/
│   ├── schema.sql
│   ├── migrations/
│   └── init.py
├── monitoring/
│   └── grafana/
└── tests/
    ├── unit/
    ├── integration/
    └── demo/
```

---

## Prerequisites

- Docker 24+
- Docker Compose v2.20+
- Python 3.11 (for local development only)

---

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd greenhouse-ai

# 2. Copy and configure environment variables
cp .env.example .env
# Edit .env — set LLM_API_KEY if using LLM explainer

# 3. Build and start all services
docker compose up --build
```

All services start on the ports listed below. Wait for all containers to report `healthy`.

---

## Service URLs

| Service | URL | Description |
|---|---|---|
| Edge 1 Gateway | http://localhost:8100 | Node 1 main API |
| Edge 1 Disease | http://localhost:8101 | ResNet-50 disease detection |
| Edge 1 Irrigation | http://localhost:8102 | XGBoost irrigation |
| Edge 1 Nutrition | http://localhost:8103 | Random Forest nutrition |
| Edge 1 Anomaly | http://localhost:8104 | Isolation Forest anomaly |
| Edge 1 Monitor | http://localhost:8105 | Model output monitor |
| Edge 1 Fusion | http://localhost:8106 | Rule fusion engine |
| Edge 1 Alarm | http://localhost:8107 | Alarm engine |
| Edge 2 Gateway | http://localhost:8200 | Node 2 main API |
| Edge 2 Disease | http://localhost:8201 | ResNet-50 disease detection |
| Edge 2 Irrigation | http://localhost:8202 | XGBoost irrigation |
| Edge 2 Nutrition | http://localhost:8203 | Random Forest nutrition |
| Edge 2 Anomaly | http://localhost:8204 | Isolation Forest anomaly |
| Edge 2 Monitor | http://localhost:8205 | Model output monitor |
| Edge 2 Fusion | http://localhost:8206 | Rule fusion engine |
| Edge 2 Alarm | http://localhost:8207 | Alarm engine |
| Central Gateway | http://localhost:9000 | Central API |
| Central Aggregator | http://localhost:9001 | Multi-node aggregator |
| Central LSTM | http://localhost:9002 | Yield forecast |
| Central LLM | http://localhost:9003 | LLM alarm explainer |
| Grafana | http://localhost:3000 | Monitoring dashboards |

---

## Development

### Run a single service

```bash
# Start only the disease model service for Node 1
docker compose up edge1_disease

# Or run locally without Docker
cd models/disease
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8101 --reload
```

### Check service health

```bash
curl http://localhost:8101/health
# {"status":"ok","service":"edge_disease","version":"0.1.0"}
```

### View API docs (OpenAPI)

Each FastAPI service exposes auto-generated docs at `/docs`:
```
http://localhost:8101/docs
```

---

## Testing

See [tests/](tests/) — placeholder for Phase 9.

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Demo scenario
python tests/demo/run_demo.py
```

---

## Documentation

- [Architecture](docs/architecture.md)
- [API Reference](docs/api-reference.md)
