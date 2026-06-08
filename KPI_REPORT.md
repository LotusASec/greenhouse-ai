# KPI Ölçüm Raporu
Tarih: 2026-06-03 11:49 UTC

## Özet

| KPI | Hedef | Ölçülen | Durum |
|-----|-------|---------|-------|
| Alarm latency (avg) | ≤ 2000ms | 119ms | ✓ |
| LLM latency (avg) | ≤ 5000ms | 11ms | ✓ |
| Monitor precision | ≥ 0.85 | 1.0000 | ✓ |
| Anomaly precision | ≥ 0.88 | 1.0000 | ✓ |

## Alarm Latency Detail
- Samples: 10
- Min: 97ms
- Max: 163ms
- Avg: 119ms
- P95: 163ms

## LLM Latency Detail
- Samples: 5
- Avg: 11ms
- Template fallback: Yes (5/5 runs)

## Monitor Precision Detail
- Normal samples: 40
- Anomaly samples: 10
- TP: 6, FP: 0, TN: 40, FN: 4
- Precision: 1.0000

## Anomaly Precision Detail
- Normal samples: 10
- Anomaly samples: 10
- TP: 10, FP: 0, TN: 10, FN: 0
- Precision: 1.0000

## Non-Automated KPIs

| KPI | Hedef | Not |
|-----|-------|-----|
| Deployment time | ≤ 10 min | `time docker compose up --build` ile ölç |
| Edge autonomous | ≥ 72h sim | Disconnect test senaryosuna bakın |
| Disease F1 | ≥ 0.85 | Test seti ile `python models/disease/train.py --eval` |
| Irrigation accuracy | ≥ 0.90 | Test seti ile `python models/irrigation/train.py --eval` |
