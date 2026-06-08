#!/bin/bash
# Wait for all key services to become healthy
# Polls /health endpoints with timeout and interval

MAX_WAIT=600
INTERVAL=5

SERVICES=(
  "edge1_gateway:8100"
  "edge1_disease:8101"
  "edge1_irrigation:8102"
  "edge1_nutrition:8103"
  "edge1_anomaly:8104"
  "edge1_output_monitor:8105"
  "edge1_fusion:8106"
  "edge1_alarm:8107"
  "central_gateway:9000"
  "grafana:3000"
  "edge1_grafana:3001"
  "edge2_grafana:3002"
)

echo "Waiting for services (max ${MAX_WAIT}s, poll every ${INTERVAL}s)..."

start_time=$(date +%s)

while true; do
  all_ok=true

  for entry in "${SERVICES[@]}"; do
    name="${entry%%:*}"
    port="${entry##*:}"

    if [ "$port" -eq 3000 ] || [ "$port" -eq 3001 ] || [ "$port" -eq 3002 ]; then
      url="http://localhost:${port}/api/health"
    else
      url="http://localhost:${port}/health"
    fi

    http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$url" 2>/dev/null)
    if [ "$http_code" != "200" ]; then
      all_ok=false
    fi
  done

  if $all_ok; then
    echo "All services healthy."
    exit 0
  fi

  elapsed=$(( $(date +%s) - start_time ))
  if [ "$elapsed" -ge "$MAX_WAIT" ]; then
    echo "Timeout after ${MAX_WAIT}s. Some services not healthy:"
    for entry in "${SERVICES[@]}"; do
      name="${entry%%:*}"
      port="${entry##*:}"
      if [ "$port" -eq 3000 ] || [ "$port" -eq 3001 ] || [ "$port" -eq 3002 ]; then
        url="http://localhost:${port}/api/health"
      else
        url="http://localhost:${port}/health"
      fi
      code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$url" 2>/dev/null)
      echo "  ${name} (${port}): HTTP ${code:-TIMEOUT}"
    done
    exit 1
  fi

  echo "  Not all ready yet (${elapsed}s elapsed), retrying in ${INTERVAL}s..."
  sleep "$INTERVAL"
done
