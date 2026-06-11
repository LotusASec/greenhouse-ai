#!/usr/bin/env bash
# 60 saniyelik canlı demo akışı — ekran kaydı için.
#
# Her iki edge node'a sırayla farklı senaryolar enjekte eder:
#   dry     → WARNING   (sarı)
#   anomaly → HIGH      (turuncu)
#   fungal  → CRITICAL  (kırmızı)
#   normal  → INFO      (mavi)
# Farklı rule_id'ler tetiklendiği için 60s alarm cooldown'ına takılmaz.
# Arka plandaki sürekli sensör akışı (1 okuma/sn) zaten devam eder.
#
# Kullanım:  bash tests/demo/live_stream_60s.sh

cd "$(dirname "$0")/../.."

echo "=== Canlı demo akışı başlıyor: $(date +%T) — kaydı şimdi başlat ==="
for s in dry anomaly fungal normal; do
  echo
  echo "--- Senaryo: $s ($(date +%T)) ---"
  for n in 1 2; do
    python3 tests/demo/inject_scenario.py --scenario "$s" --node "$n" 2>&1 \
      | grep -E "Running|Actual level|Alarm ID" | sed "s/^/  node$n /"
  done
  [ "$s" != "normal" ] && sleep 12
done
echo
echo "=== Bitti: $(date +%T) — alarmlar ~30 sn içinde central'a da düşer ==="
