# Greenhouse AI — Service Quick Reference

## Endpoint Tables

Her endpoint için: Metod · Yol · Açıklama · Örnek curl

---

## Edge Node 1 — Gateway `:8100`

> Edge 2 için tüm portları 8100→8200, 8101→8201, ... şeklinde değiştirin.

| Metod | Yol | Açıklama |
|-------|-----|----------|
| GET | `/health` | 7 downstream servis durumu (hastalık, sulama, beslenme, anomali, monitor, fusion, alarm) |
| POST | `/predict` | Tek tam pipeline: sensör+görüntü → model çağrıları → alarm |
| GET | `/status` | Sürekli döngü istatistikleri + servis URL'leri |
| GET | `/threshold` | Fusion Engine kural listesini proxy eder |
| PUT | `/threshold/{rule_id}` | Kural eşiği güncelle (Fusion Engine'e proxy) |
| GET | `/logs` | SQLite sensor_readings tablosundan son okumalar |
| GET | `/alerts` | Alarm Engine'den filtrelenmiş alarm listesi |
| GET | `/sync` | Senkronize edilmemiş alarmlar (central için alias) |
| GET | `/alerts/unsynced` | `/sync` ile aynı — synced=0 alarmlar |
| POST | `/alerts/sync-all` | Tüm unsynced alarmları synced=1 yap |

```bash
# Sağlık kontrolü — tüm 7 downstream servis
curl http://localhost:8100/health

# Tek tam pipeline çalıştır (görüntüsüz)
curl -X POST http://localhost:8100/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_reading": {
      "node_id": "greenhouse_01",
      "timestamp": "2024-01-15T10:00:00Z",
      "sensors": {
        "temperature": 24.0,
        "humidity": 65.0,
        "soil_moisture": 55.0,
        "light": 700.0,
        "ec": 2.2,
        "ph": 6.3
      }
    }
  }'

# Fungal alarm senaryosu (RULE_009 — CRITICAL)
curl -X POST http://localhost:8100/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_reading": {
      "node_id": "greenhouse_01",
      "timestamp": "2024-01-15T10:00:00Z",
      "sensors": {
        "temperature": 22.0,
        "humidity": 88.0,
        "soil_moisture": 70.0,
        "light": 400.0,
        "ec": 1.2,
        "ph": 7.5
      }
    }
  }'

# Kural listesi
curl http://localhost:8100/threshold

# Kural eşiği değiştir (RULE_001 humidity threshold → 75)
curl -X PUT http://localhost:8100/threshold/RULE_001 \
  -H "Content-Type: application/json" \
  -d '{"field": "conditions[0].value", "value": 75}'

# Alarm geçmişi (son 20, sadece CRITICAL)
curl "http://localhost:8100/alerts?level=CRITICAL&limit=20"

# Senkronize edilmemiş alarmlar
curl http://localhost:8100/alerts/unsynced

# Tüm alarmları senkronize işaretle
curl -X POST http://localhost:8100/alerts/sync-all

# Döngü istatistikleri
curl http://localhost:8100/status

# Son 50 sensör okuması
curl "http://localhost:8100/logs?limit=50"
```

---

## Disease Model `:8101`

| Metod | Yol | Açıklama |
|-------|-----|----------|
| GET | `/health` | Model yüklü mü; model dosya yolunu döner |
| POST | `/predict` | base64 görüntü → hastalık sınıfı + olasılıklar |

**İstek şeması** (`POST /predict`):
```json
{
  "node_id": "greenhouse_01",
  "timestamp": "2024-01-15T10:00:00Z",
  "image_b64": "<base64_jpeg_string>"
}
```

**Yanıt şeması** (MASTER_SPEC §3.2):
```json
{
  "node_id": "greenhouse_01",
  "timestamp": "2024-01-15T10:00:00Z",
  "top_prediction": "healthy",
  "top_confidence": 0.923,
  "class_probabilities": {
    "healthy": 0.923,
    "early_blight": 0.031,
    "late_blight": 0.024,
    "leaf_mold": 0.012,
    "other": 0.010
  }
}
```

```bash
# Sağlık kontrolü
curl http://localhost:8101/health

# Görüntü ile tahmin (örnek: test görüntüsünü base64'e çevir)
IMAGE_B64=$(base64 -w0 /path/to/test_image.jpg)
curl -X POST http://localhost:8101/predict \
  -H "Content-Type: application/json" \
  -d "{\"node_id\": \"greenhouse_01\", \"timestamp\": \"2024-01-15T10:00:00Z\", \"image_b64\": \"$IMAGE_B64\"}"
```

---

## Irrigation Model `:8102`

| Metod | Yol | Açıklama |
|-------|-----|----------|
| GET | `/health` | Her iki model yüklü mü (classifier + regressor) |
| POST | `/predict` | Sensör değerleri → sulama kararı + miktar |

**İstek şeması:**
```json
{
  "node_id": "greenhouse_01",
  "timestamp": "2024-01-15T10:00:00Z",
  "temperature": 24.0,
  "humidity": 65.0,
  "soil_moisture": 55.0,
  "light": 700.0,
  "ec": 2.2,
  "ph": 6.3
}
```

**Yanıt şeması** (MASTER_SPEC §3.3):
```json
{
  "node_id": "greenhouse_01",
  "timestamp": "2024-01-15T10:00:00Z",
  "irrigate": false,
  "amount_liters": 0.0,
  "confidence": 0.87
}
```

```bash
curl http://localhost:8102/health

# Kuru toprak — sulama gerekli
curl -X POST http://localhost:8102/predict \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "greenhouse_01",
    "timestamp": "2024-01-15T10:00:00Z",
    "temperature": 28.0,
    "humidity": 45.0,
    "soil_moisture": 22.0,
    "light": 800.0,
    "ec": 2.0,
    "ph": 6.5
  }'
```

---

## Nutrition Model `:8103`

| Metod | Yol | Açıklama |
|-------|-----|----------|
| GET | `/health` | Model yüklü mü |
| POST | `/predict` | Sensör değerleri → besin eksikliği sınıfı |

**İstek şeması:** Irrigation ile aynı (6 sensör değeri + node_id + timestamp)

**Yanıt şeması** (MASTER_SPEC §3.4):
```json
{
  "node_id": "greenhouse_01",
  "timestamp": "2024-01-15T10:00:00Z",
  "deficiency_class": "normal",
  "fertilizer_recommendation": "Beslenme dengeli, mevcut gübre programına devam edin.",
  "confidence": 0.91
}
```

Olası sınıflar: `normal` | `N_deficiency` | `P_deficiency` | `K_deficiency`

```bash
curl http://localhost:8103/health

curl -X POST http://localhost:8103/predict \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "greenhouse_01",
    "timestamp": "2024-01-15T10:00:00Z",
    "temperature": 24.0,
    "humidity": 65.0,
    "soil_moisture": 55.0,
    "light": 700.0,
    "ec": 1.0,
    "ph": 6.3
  }'
```

---

## Anomaly Model `:8104`

| Metod | Yol | Açıklama |
|-------|-----|----------|
| GET | `/health` | Model yüklü mü |
| POST | `/predict` | Sensör değerleri → anomali skoru |

**Yanıt şeması** (MASTER_SPEC §3.5):
```json
{
  "node_id": "greenhouse_01",
  "timestamp": "2024-01-15T10:00:00Z",
  "is_anomaly": false,
  "anomaly_score": 0.12
}
```

`anomaly_score`: 0.0 (tamamen normal) → 1.0 (güçlü anomali); sigmoid(raw × 5)

```bash
curl http://localhost:8104/health

# Ekstrem değerler → anomali beklenir
curl -X POST http://localhost:8104/predict \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "greenhouse_01",
    "timestamp": "2024-01-15T10:00:00Z",
    "temperature": 48.0,
    "humidity": 99.0,
    "soil_moisture": 98.0,
    "light": 1950.0,
    "ec": 4.8,
    "ph": 1.2
  }'
```

---

## Output Monitor `:8105`

| Metod | Yol | Açıklama |
|-------|-----|----------|
| GET | `/health` | Servis sağlık |
| POST | `/monitor/{model_name}` | Tek metrik değer gönder; z-score + IQR analizi |
| GET | `/status` | Tüm 4 modelin buffer istatistikleri |
| GET | `/status/{model_name}` | Tek modelin buffer istatistiği |

`model_name` ∈ `{disease, irrigation, nutrition, anomaly}`

**İstek şeması** (`POST /monitor/{model_name}`):
```json
{
  "node_id": "greenhouse_01",
  "timestamp": "2024-01-15T10:00:00Z",
  "metric_value": 0.923
}
```

**Yanıt şeması** (MASTER_SPEC §3.6):
```json
{
  "node_id": "greenhouse_01",
  "timestamp": "2024-01-15T10:00:00Z",
  "model_name": "disease",
  "metric_value": 0.923,
  "is_anomaly": false,
  "z_score": 0.45,
  "window_mean": 0.89,
  "window_std": 0.07,
  "window_size": 38
}
```

```bash
curl http://localhost:8105/health

# Disease confidence değeri gönder
curl -X POST http://localhost:8105/monitor/disease \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "greenhouse_01",
    "timestamp": "2024-01-15T10:00:00Z",
    "metric_value": 0.923
  }'

# Drift testi: 50 normal değer → buffer doldur
for i in $(seq 1 50); do
  curl -s -X POST http://localhost:8105/monitor/disease \
    -H "Content-Type: application/json" \
    -d "{\"node_id\": \"greenhouse_01\", \"timestamp\": \"2024-01-15T10:00:00Z\", \"metric_value\": 0.45}" > /dev/null
done
# Spike değer → is_anomaly beklenir
curl -X POST http://localhost:8105/monitor/disease \
  -H "Content-Type: application/json" \
  -d '{"node_id": "greenhouse_01", "timestamp": "2024-01-15T10:00:00Z", "metric_value": 0.10}'

# Tüm buffer durumları
curl http://localhost:8105/status

# Sadece disease buffer
curl http://localhost:8105/status/disease
```

---

## Fusion Engine `:8106`

| Metod | Yol | Açıklama |
|-------|-----|----------|
| GET | `/health` | Yüklenen kural sayısı ile birlikte durum |
| POST | `/evaluate` | FusionInput → AlarmObject (kural değerlendirme + DB yazma) |
| GET | `/rules` | 10 kuralın tamamı JSON olarak |
| PUT | `/rules/{rule_id}/threshold` | Kural eşiği güncelleme (bellekte + rules.yaml) |

**İstek şeması** (`POST /evaluate`, MASTER_SPEC §3.8):
```json
{
  "sensor_reading": {
    "node_id": "greenhouse_01",
    "timestamp": "2024-01-15T10:00:00Z",
    "sensors": {"temperature": 24.0, "humidity": 88.0, "soil_moisture": 70.0, "light": 400.0, "ec": 1.2, "ph": 7.5}
  },
  "disease_output": {"top_prediction": "healthy", "top_confidence": 0.45, "class_probabilities": {}},
  "irrigation_output": {"irrigate": false, "amount_liters": 0.0, "confidence": 0.8},
  "nutrition_output": {"deficiency_class": "N_deficiency", "confidence": 0.85},
  "anomaly_output": {"is_anomaly": false, "anomaly_score": 0.1},
  "monitor_events": []
}
```

**Yanıt şeması** (AlarmObject, MASTER_SPEC §3.7):
```json
{
  "alarm_id": "ALM_20240115_A3F9E2",
  "node_id": "greenhouse_01",
  "timestamp": "2024-01-15T10:00:00Z",
  "level": "CRITICAL",
  "source": "rule_engine",
  "rule_id": "RULE_009",
  "message_key": "fungal_risk_critical",
  "trigger_values": {"humidity": 88.0, "ec": 1.2, "ph": 7.5}
}
```

```bash
curl http://localhost:8106/health

# Tüm kuralları listele
curl http://localhost:8106/rules | python3 -m json.tool

# Kural eşiğini değiştir (RULE_001 humidity koşulu → 85)
curl -X PUT http://localhost:8106/rules/RULE_001/threshold \
  -H "Content-Type: application/json" \
  -d '{"field": "conditions[0].value", "value": 85}'
```

---

## Alarm Engine `:8107`

| Metod | Yol | Açıklama |
|-------|-----|----------|
| GET | `/health` | Servis + DB bağlantısı |
| POST | `/alarm` | Alarm depola (cooldown kontrolü ile) |
| GET | `/alerts` | Filtrelenmiş alarm listesi (`level`, `source`, `since`, `limit`, `offset`) |
| GET | `/alerts/{alarm_id}` | Tek alarm |
| GET | `/alerts/summary` | Level + source bazlı sayım |
| GET | `/alerts/unsynced` | `synced=0` alarmlar |
| POST | `/alerts/{alarm_id}/sync` | Tek alarmı `synced=1` yap |
| POST | `/alerts/sync-all` | Tüm unsynced alarmları `synced=1` yap |

```bash
curl http://localhost:8107/health

# Alarm geçmişi (level ve tarih filtresi)
curl "http://localhost:8107/alerts?level=CRITICAL&since=2024-01-15T00:00:00Z&limit=10"

# Sayfalama
curl "http://localhost:8107/alerts?limit=20&offset=40"

# Kaynak bazlı filtre (rule_engine | model_monitor)
curl "http://localhost:8107/alerts?source=model_monitor"

# Özet
curl http://localhost:8107/alerts/summary

# Unsynced alarmlar
curl http://localhost:8107/alerts/unsynced

# Tek alarm detayı
curl http://localhost:8107/alerts/ALM_20240115_A3F9E2

# Tek alarmı işaretle
curl -X POST http://localhost:8107/alerts/ALM_20240115_A3F9E2/sync

# Tümünü işaretle
curl -X POST http://localhost:8107/alerts/sync-all
```

---

## Central Gateway `:9000`

> Tüm merkez servisler için tek giriş noktası. Kendi iş mantığı yok, pure proxy.

| Metod | Yol | Proxy Hedefi | Açıklama |
|-------|-----|--------------|----------|
| GET | `/health` | AGG + LSTM + LLM | 3 servis sağlık |
| GET | `/status` | AGG:9001/status | Sync durumu |
| GET | `/nodes` | AGG:9001/nodes | Kayıtlı node listesi |
| GET | `/nodes/{node_id}/status` | AGG | Node bağlantı durumu |
| GET | `/nodes/{node_id}/alerts` | AGG | Node'a ait alarmlar |
| POST | `/nodes/register` | AGG | Yeni node kaydet |
| POST | `/nodes/{node_id}/sync` | AGG | Elle senkronizasyon tetikle |
| GET | `/alerts` | AGG:9001/alerts | Filtrelenmiş birleşik alarmlar |
| GET | `/alerts/summary` | AGG | Level/source sayımı |
| GET | `/yield` | LSTM:9002/yield | Tüm node'lar verim tahmini |
| GET | `/yield/{node_id}` | LSTM | Tek node verim tahmini |
| POST | `/explain` | LLM:9003/explain | Alarm → Türkçe açıklama |

```bash
curl http://localhost:9000/health

# Kayıtlı node'lar
curl http://localhost:9000/nodes | python3 -m json.tool

# Bir node'un durumu
curl http://localhost:9000/nodes/greenhouse_01/status

# Tüm alarmlar (merkez DB)
curl "http://localhost:9000/alerts?limit=50"

# Alarm özeti
curl http://localhost:9000/alerts/summary

# Verim tahminleri
curl http://localhost:9000/yield | python3 -m json.tool

# Tek node verim
curl http://localhost:9000/yield/greenhouse_01

# Elle sync tetikle
curl -X POST http://localhost:9000/nodes/greenhouse_01/sync
```

---

## Central Aggregator `:9001`

(Central Gateway üzerinden ulaşılması önerilir; doğrudan erişim için aşağıdaki portları kullanın)

| Metod | Yol | Açıklama |
|-------|-----|----------|
| GET | `/health` | Servis + DB |
| GET | `/nodes` | Kayıtlı node listesi + metadata |
| GET | `/nodes/{node_id}/status` | Son ping, son sync, alarm sayısı |
| GET | `/nodes/{node_id}/alerts` | Node'a ait alarmlar (merkez DB'den) |
| POST | `/nodes/register` | `{ node_id, gateway_url }` ile node kaydet |
| POST | `/nodes/{node_id}/sync` | O node ile anında senkronizasyon |
| GET | `/alerts` | Filtrelenmiş merkez alarm listesi |
| GET | `/alerts/summary` | Level + source + node bazlı sayım |
| GET | `/status` | Sync loop durumu + tüm node metadata |
| POST | `/alarms/{alarm_id}/explain` | `{ explanation }` body ile DB güncelle |

```bash
curl http://localhost:9001/health

# Yeni node kaydet
curl -X POST http://localhost:9001/nodes/register \
  -H "Content-Type: application/json" \
  -d '{"node_id": "greenhouse_03", "gateway_url": "http://edge3_gateway:8100"}'

# greenhouse_01 anında senkronize et
curl -X POST http://localhost:9001/nodes/greenhouse_01/sync

# LLM açıklamasını DB'ye kaydet
curl -X POST http://localhost:9001/alarms/ALM_20240115_A3F9E2/explain \
  -H "Content-Type: application/json" \
  -d '{"explanation": "Yüksek nem ve düşük EC değeri fungal hastalık riskini artırmaktadır."}'
```

---

## Central LSTM Yield `:9002`

| Metod | Yol | Açıklama |
|-------|-----|----------|
| GET | `/health` | Model dosyası yüklü mü (`model_ready: bool`) |
| GET | `/yield/{node_id}` | Tek node için 7-günlük LSTM verim tahmini |
| GET | `/yield` | `greenhouse_01` ve `greenhouse_02` için tahminler |

**Yanıt şeması** (normal durum):
```json
{
  "node_id": "greenhouse_01",
  "status": "ok",
  "yield_score": 0.73,
  "confidence": "high",
  "days_used": 35,
  "forecasted_at": "2024-01-15T10:00:00Z"
}
```

**Yetersiz veri:**
```json
{
  "node_id": "greenhouse_01",
  "status": "insufficient_data",
  "min_days_required": 14,
  "days_available": 7
}
```

**Model eğitilmemiş:**
```json
{
  "node_id": "greenhouse_01",
  "status": "model_not_trained"
}
```

`confidence`: `low` (<21 gün) | `medium` (21-30 gün) | `high` (>30 gün)

```bash
curl http://localhost:9002/health

curl http://localhost:9002/yield/greenhouse_01
curl http://localhost:9002/yield/greenhouse_02
curl http://localhost:9002/yield
```

---

## Central LLM Explainer `:9003`

| Metod | Yol | Açıklama |
|-------|-----|----------|
| GET | `/health` | `llm_available: bool` — OpenAI API erişilebilir mi |
| POST | `/explain` | AlarmObject → Türkçe açıklama + DB güncelleme |

**İstek şeması:**
```json
{
  "alarm_id": "ALM_20240115_A3F9E2",
  "node_id": "greenhouse_01",
  "timestamp": "2024-01-15T10:00:00Z",
  "level": "CRITICAL",
  "source": "rule_engine",
  "rule_id": "RULE_009",
  "message_key": "fungal_risk_critical",
  "trigger_values": {"humidity": 88.0, "ec": 1.2}
}
```

**Yanıt şeması:**
```json
{
  "alarm_id": "ALM_20240115_A3F9E2",
  "explanation": "Yüksek nem (%88) ve düşük EC değeri (1.2 mS/cm) fungal hastalık için uygun ortam oluşturmaktadır. İmidakloprid bazlı mantar ilacı uygulaması ve havalandırma artırımı önerilir. Nem seviyesini %70 altına indirmek kritik öncelik.",
  "source": "llm",
  "model_used": "gpt-4o-mini"
}
```

`source`: `"llm"` (OpenAI API kullanıldı) | `"template"` (fallback kullanıldı)

```bash
curl http://localhost:9003/health

# Alarm açıkla
curl -X POST http://localhost:9003/explain \
  -H "Content-Type: application/json" \
  -d '{
    "alarm_id": "ALM_20240115_A3F9E2",
    "node_id": "greenhouse_01",
    "timestamp": "2024-01-15T10:00:00Z",
    "level": "CRITICAL",
    "source": "rule_engine",
    "rule_id": "RULE_009",
    "message_key": "fungal_risk_critical",
    "trigger_values": {"humidity": 88.0, "ec": 1.2}
  }'
```

---

## Grafana

| Instance | Port | Datasource | Dashboard |
|----------|------|------------|-----------|
| Central | 3000 | CentralDB (`/data/central.db`) | central.json |
| Edge Node 1 | 3001 | EdgeDB (`/data/greenhouse.db` — edge1) | edge_node.json |
| Edge Node 2 | 3002 | EdgeDB (`/data/greenhouse.db` — edge2) | edge_node.json |

- Tüm instanceler için kullanıcı adı: `admin`, şifre: `greenhouse`
- URL: `http://localhost:<port>`

```bash
# Grafana sağlık (tüm 3 instance)
curl http://localhost:3000/api/health
curl http://localhost:3001/api/health
curl http://localhost:3002/api/health

# CentralDB datasource doğrulama
curl -u admin:greenhouse http://localhost:3000/api/datasources

# EdgeDB datasource (edge1)
curl -u admin:greenhouse http://localhost:3001/api/datasources

# Dashboard listesi
curl -u admin:greenhouse http://localhost:3000/api/search?type=dash-db

# SQLite datasource üzerinden doğrudan sorgu (central)
curl -u admin:greenhouse -X POST http://localhost:3000/api/ds/query \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [{
      "refId": "A",
      "datasource": {"uid": "CentralDB"},
      "rawSQL": "SELECT COUNT(*) as total, level FROM alarms GROUP BY level",
      "format": "table"
    }],
    "from": "now-24h",
    "to": "now"
  }'
```

---

## Örnek Curl Senaryoları

### Senaryo 1: Normal Çalışma Doğrulama

```bash
# 1. Tüm servisler ayakta mı?
curl -s http://localhost:8100/health | python3 -m json.tool

# 2. Normal sensör verisi — INFO alarm beklenir
curl -s -X POST http://localhost:8100/predict \
  -H "Content-Type: application/json" \
  -d '{"sensor_reading": {"node_id": "greenhouse_01", "timestamp": "2024-01-15T10:00:00Z", "sensors": {"temperature": 24.0, "humidity": 65.0, "soil_moisture": 55.0, "light": 700.0, "ec": 2.2, "ph": 6.3}}}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Level: {d[\"level\"]}, Rule: {d.get(\"rule_id\",\"-\")}')"
```

### Senaryo 2: Fungal Risk Tetikleme (RULE_009 — CRITICAL)

```bash
curl -s -X POST http://localhost:8100/predict \
  -H "Content-Type: application/json" \
  -d '{"sensor_reading": {"node_id": "greenhouse_01", "timestamp": "2024-01-15T10:00:00Z", "sensors": {"temperature": 22.0, "humidity": 88.0, "soil_moisture": 70.0, "light": 400.0, "ec": 1.2, "ph": 7.5}}}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Level: {d[\"level\"]}, Rule: {d[\"rule_id\"]}, Alarm: {d[\"alarm_id\"]}')"
```

### Senaryo 3: Alarm Geçmişi + LLM Açıklama

```bash
# Son alarmlara bak
ALARM_ID=$(curl -s http://localhost:9000/alerts?limit=1 | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['alarm_id']) if d else print('')")

# Varsa LLM ile açıkla
if [ -n "$ALARM_ID" ]; then
  curl -s -X POST http://localhost:9000/explain \
    -H "Content-Type: application/json" \
    -d "$(curl -s http://localhost:9001/alerts | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d[0])) if d else print('{}')")"
fi
```

### Senaryo 4: Merkez Node Durumu

```bash
# Tüm node'lar
curl -s http://localhost:9000/nodes | python3 -m json.tool

# Verim tahminleri
curl -s http://localhost:9000/yield | python3 -m json.tool

# Merkez alarmları (son 24 saat)
curl -s "http://localhost:9000/alerts?since=$(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%SZ)&limit=100" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Toplam: {len(d)} alarm')"
```

### Senaryo 5: Model Drift Tespiti

```bash
# Önce 50 kararlı değer ile buffer doldur
for i in $(seq 1 50); do
  curl -s -X POST http://localhost:8105/monitor/disease \
    -H "Content-Type: application/json" \
    -d '{"node_id": "greenhouse_01", "timestamp": "2024-01-15T10:00:00Z", "metric_value": 0.45}' > /dev/null
done

# Spike → is_anomaly=true beklenir
curl -s -X POST http://localhost:8105/monitor/disease \
  -H "Content-Type: application/json" \
  -d '{"node_id": "greenhouse_01", "timestamp": "2024-01-15T10:00:01Z", "metric_value": 0.10}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'is_anomaly: {d[\"is_anomaly\"]}, z_score: {d[\"z_score\"]:.2f}')"
```

### Senaryo 6: El ile Senkronizasyon

```bash
# Edge'deki unsynced alarmları gör
curl http://localhost:8100/alerts/unsynced | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d)} unsynced alarm')"

# El ile sync tetikle
curl -X POST http://localhost:9000/nodes/greenhouse_01/sync

# Doğrula
curl http://localhost:8100/alerts/unsynced | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d)} unsynced alarm kaldı')"
```

---

## Hızlı Kontrol Listesi — İlk Kurulum

```bash
# 1. Servislerin ayakta olduğunu doğrula
./scripts/health_check.sh

# 2. Grafana datasource kontrol
curl -u admin:greenhouse http://localhost:3000/api/datasources | python3 -c "import sys,json; [print(d['name'], d['type']) for d in json.load(sys.stdin)]"

# 3. Node'ların merkez tarafından görüldüğünü doğrula
curl http://localhost:9000/nodes

# 4. İlk alarm üret
curl -s -X POST http://localhost:8100/predict \
  -H "Content-Type: application/json" \
  -d '{"sensor_reading": {"node_id": "greenhouse_01", "timestamp": "2024-01-15T10:00:00Z", "sensors": {"temperature": 22.0, "humidity": 88.0, "soil_moisture": 70.0, "light": 400.0, "ec": 1.2, "ph": 7.5}}}'

# 5. Merkez'e ulaştığını kontrol et (30s sync)
sleep 35 && curl http://localhost:9000/alerts/summary
```
