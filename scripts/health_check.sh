#!/bin/bash
# Quick health check — no waiting, prints service table, exits 1 if any unhealthy

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
  "central_aggregator:9001"
  "central_lstm:9002"
  "central_llm:9003"
  "grafana:3000"
  "edge1_grafana:3001"
  "edge2_grafana:3002"
)

printf "%-30s %-8s %-10s\n" "SERVICE" "PORT" "STATUS"
printf "%-30s %-8s %-10s\n" "-------" "----" "------"

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

  if [ "$http_code" = "200" ]; then
    status="OK"
  else
    status="FAIL (${http_code:-TIMEOUT})"
    all_ok=false
  fi

  printf "%-30s %-8s %-10s\n" "$name" "$port" "$status"
done

if $all_ok; then
  echo ""
  echo "All services healthy."
  exit 0
else
  echo ""
  echo "One or more services unhealthy."
  exit 1
fi
