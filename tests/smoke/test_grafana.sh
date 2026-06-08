#!/usr/bin/env bash
# Greenhouse AI — Grafana smoke test
# Usage: bash tests/smoke/test_grafana.sh

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'

ok()   { echo -e "${GREEN}[OK]${RESET}  $*"; }
fail() { echo -e "${RED}[FAIL]${RESET} $*"; FAILURES=$((FAILURES + 1)); }
info() { echo -e "${YELLOW}[INFO]${RESET} $*"; }

FAILURES=0

echo ""
echo "=== Greenhouse AI — Grafana Smoke Test ==="
echo ""

check_health() {
  local label="$1"
  local url="$2"
  local status
  status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$url" 2>/dev/null || echo "000")
  if [[ "$status" == "200" ]]; then
    ok "$label → HTTP $status"
  else
    fail "$label → HTTP $status (expected 200)"
  fi
}

# 1. Health checks
info "Checking Grafana health endpoints..."
check_health "Central Grafana  :3000" "http://localhost:3000/api/health"
check_health "Edge1 Grafana    :3001" "http://localhost:3001/api/health"
check_health "Edge2 Grafana    :3002" "http://localhost:3002/api/health"

echo ""

# 2. Datasource check (central)
info "Checking datasources on central Grafana..."
ds_resp=$(curl -s -u admin:greenhouse \
  "http://localhost:3000/api/datasources" --max-time 5 2>/dev/null || echo "[]")
if echo "$ds_resp" | python3 -c "import sys,json; ds=json.load(sys.stdin); names=[d.get('name') for d in ds]; exit(0 if 'CentralDB' in names else 1)" 2>/dev/null; then
  ok "CentralDB datasource provisioned"
else
  fail "CentralDB datasource not found in: $ds_resp"
fi

echo ""

# 3. Dashboard list
info "Checking dashboards on central Grafana..."
dash_resp=$(curl -s -u admin:greenhouse \
  "http://localhost:3000/api/search?type=dash-db" --max-time 5 2>/dev/null || echo "[]")
dash_count=$(echo "$dash_resp" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
if [[ "$dash_count" -ge 2 ]]; then
  ok "Dashboards loaded: $dash_count found"
  echo "$dash_resp" | python3 -c "import sys,json; [print('  -', d.get('title','?')) for d in json.load(sys.stdin)]" 2>/dev/null || true
else
  fail "Expected ≥2 dashboards, got $dash_count"
  echo "  Response: $dash_resp"
fi

echo ""

# 4. SQLite datasource query test
info "Testing SQLite datasource query (central)..."
# Get datasource UID
ds_uid=$(curl -s -u admin:greenhouse "http://localhost:3000/api/datasources" --max-time 5 2>/dev/null | \
  python3 -c "import sys,json; ds=[d for d in json.load(sys.stdin) if d.get('name')=='CentralDB']; print(ds[0].get('uid','') if ds else '')" 2>/dev/null || echo "")

if [[ -n "$ds_uid" ]]; then
  query_resp=$(curl -s -u admin:greenhouse \
    -X POST "http://localhost:3000/api/ds/query" \
    -H "Content-Type: application/json" \
    -d "{\"queries\":[{\"datasource\":{\"type\":\"frser-sqlite-datasource\",\"uid\":\"$ds_uid\"},\"rawSql\":\"SELECT COUNT(*) as count FROM alarms\",\"format\":\"table\",\"refId\":\"A\"}],\"from\":\"now-1h\",\"to\":\"now\"}" \
    --max-time 10 2>/dev/null || echo "{}")
  if echo "$query_resp" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('results') else 1)" 2>/dev/null; then
    ok "SQLite query executed successfully"
  else
    fail "SQLite query failed: ${query_resp:0:100}"
  fi
else
  info "Skipping SQLite query test (datasource UID not found)"
fi

echo ""
echo "=== Results ==="
if [[ $FAILURES -eq 0 ]]; then
  echo -e "${GREEN}All checks passed!${RESET}"
else
  echo -e "${RED}$FAILURES check(s) failed.${RESET}"
  exit 1
fi
