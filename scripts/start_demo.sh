#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Greenhouse AI — Demo Starting"
echo "Working directory: $PROJECT_DIR"

cd "$PROJECT_DIR"

# 1. Start all services (idempotent)
echo "Starting services..."
docker compose up -d

# 2. Wait for services to become healthy (max 10 minutes)
echo "Waiting for services to become healthy..."
"$SCRIPT_DIR/wait_for_services.sh"

# 3. Seed historical data for LSTM (best-effort)
echo "Seeding LSTM historical data..."
docker compose exec -T central_lstm python scripts/seed_historical_data.py \
  --db-path /data/central.db 2>/dev/null || echo "  (seed skipped — service not ready or script not found)"

# 4. Final health check
echo "Final health check..."
"$SCRIPT_DIR/health_check.sh"

echo ""
echo "System ready!"
echo "  Central Grafana:  http://localhost:3000"
echo "  Edge 1 Grafana:   http://localhost:3001"
echo "  Edge 2 Grafana:   http://localhost:3002"
echo "  Edge API:         http://localhost:8100"
echo "  Central API:      http://localhost:9000"
