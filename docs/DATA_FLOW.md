# Greenhouse AI — Veri Akışı Dokümantasyonu

## Mutlu Yol (Happy Path): Sensörden Merkez Alarma

Aşağıdaki tablo, bir sensör okumasının sisteme girişinden merkez DB'ye kaydına ve Grafana'da görünmesine kadar geçen her adımı göstermektedir.

---

### Adım 1 — Sensör Üretimi (Simulator)

| Öğe | Değer |
|-----|-------|
| Kaynak kod | `simulator/sensor_simulator.py` — `SensorSimulator.generate_reading()` (satır ~45) |
| Hangi servis çalıştırır | `edge1_gateway` → `EdgePipeline.__init__` içinde `SensorSimulator` başlatılır |
| DB tablosu | `sensor_readings` |
| DB dosyası | `/data/greenhouse.db` (edge1) veya `/data/greenhouse.db` (edge2, ayrı Docker volume) |
| Tetikleyen kod | `EdgePipeline.start_continuous_loop()` → `simulator.generate_reading()` → `simulator._write_db(reading)` |

**Üretilen kayıt yapısı:**
```
sensor_readings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  node_id TEXT,           -- "greenhouse_01"
  timestamp TEXT,         -- ISO 8601 UTC
  temperature REAL,
  humidity REAL,
  soil_moisture REAL,
  light REAL,
  ec REAL,
  ph REAL
)
```

**Tetikleme frekansı:** `emit_interval_seconds` (node YAML, varsayılan 1 saniye)

---

### Adım 2 — Model Çağrıları (4 Model Paralel)

| Öğe | Değer |
|-----|-------|
| Kaynak kod | `edge/gateway/pipeline.py` — `EdgePipeline.run_inference()` (satır ~80) |
| Paralel çağrı | `asyncio.gather(return_exceptions=True)` — 4 model aynı anda çağrılır |
| Timeout | Her model için 10 saniye |
| Fallback | `_get_fallback(model_name, node_id, timestamp)` — model çöktüğünde nötr değer |

**Model → Endpoint → Yanıt:**

| Model | URL | Kritik Çıktı |
|-------|-----|--------------|
| Disease | `http://edge1_disease:8101/predict` | `top_prediction`, `top_confidence`, `class_probabilities` |
| Irrigation | `http://edge1_irrigation:8102/predict` | `irrigate`, `amount_liters`, `confidence` |
| Nutrition | `http://edge1_nutrition:8103/predict` | `deficiency_class`, `confidence` |
| Anomaly | `http://edge1_anomaly:8104/predict` | `is_anomaly`, `anomaly_score` |

**Fallback değerleri (model çökerse):**
```python
disease:    {"top_prediction": "unknown", "top_confidence": 0.0, "class_probabilities": {}}
irrigation: {"irrigate": False, "amount_liters": 0.0, "confidence": 0.0}
nutrition:  {"deficiency_class": "normal", "confidence": 0.0}
anomaly:    {"is_anomaly": False, "anomaly_score": 0.0}
```

**Önemli**: `DATASET_PATH` env değişkeni boşsa ImageSimulator başlamaz → `image_b64=None` → disease modeli her zaman fallback değer döner.

---

### Adım 3 — Output Monitor Kontrolü

| Öğe | Değer |
|-----|-------|
| Kaynak kod | `edge/gateway/pipeline.py` — `EdgePipeline._call_monitor_all()` (satır ~130) |
| Çağrılan endpoint | `POST http://edge1_output_monitor:8105/monitor/{model_name}` |
| Monitörlenen değer | disease → `top_confidence`; irrigation → `confidence`; nutrition → `confidence`; anomaly → `anomaly_score` |

**Monitor işlem akışı** (`models/output_monitor/monitor.py`):
1. `MonitorService.check(model_name, value, node_id, timestamp)` çağrılır
2. `deque[maxlen=50]` buffer'a değer eklenir
3. `window_size < MIN_SAMPLES (10)` ise `is_anomaly=False` (ısınma periyodu)
4. `z_score = (value - mean) / (std + 1e-9)` hesaplanır
5. `IQR = Q3 - Q1`; değer `Q1 - 1.5*IQR` altında veya `Q3 + 1.5*IQR` üstündeyse IQR anomali
6. `is_anomaly = z_score > Z_THRESHOLD(2.5) AND iqr_anomaly` (ikisi de True olmalı)
7. `DB_PATH` varsa `monitor_events` tablosuna yazar

**DB tablosu:**
```
monitor_events (
  id INTEGER PRIMARY KEY,
  node_id TEXT,
  timestamp TEXT,
  model_name TEXT,
  metric_value REAL,
  z_score REAL,
  window_mean REAL,
  window_std REAL,
  is_anomaly INTEGER      -- 0 veya 1
)
```

---

### Adım 4 — Fusion Input Oluşturma

| Öğe | Değer |
|-----|-------|
| Kaynak kod | `edge/gateway/pipeline.py` — `EdgePipeline._build_fusion_input()` (satır ~160) |
| Girdi | 4 model yanıtı + monitor olayları + orijinal sensör okuması |
| Çıktı | MASTER_SPEC §3.8 formatında `FusionInput` dict |

**FusionInput yapısı:**
```json
{
  "sensor_reading": { "node_id": "...", "timestamp": "...", "sensors": {...} },
  "disease_output":    { ... },
  "irrigation_output": { ... },
  "nutrition_output":  { ... },
  "anomaly_output":    { ... },
  "monitor_events":    [ { "model_name": "disease", "is_anomaly": false, ... } ]
}
```

---

### Adım 5 — Kural Motoru Değerlendirmesi (Fusion Engine)

| Öğe | Değer |
|-----|-------|
| Kaynak kod | `edge/fusion_engine/engine.py` — `FusionEngine.process(fusion_input)` (satır ~85) |
| Çağrılan endpoint | `POST http://edge1_fusion:8106/evaluate` |
| Kural dosyası | `edge/fusion_engine/rules/rules.yaml` |

**Kural değerlendirme akışı** (`RuleEngine.evaluate`):
1. 10 kural sırayla denenir (RULE_001 → RULE_010)
2. Her kural için tüm `conditions[]` `AND` mantığıyla değerlendirilir
3. `_get_nested(fusion_input, field_path)` ile noktalı path gezilir (örn: `sensor_reading.sensors.humidity`)
4. İlk eşleşen kural döner; eşleşme yoksa `{ level: "INFO", rule_id: "no_rule_matched" }`

**Monitor birleştirme** (`FusionEngine._merge_results`):
- Kural sonucu ve monitor olayları karşılaştırılır
- Yüksek seviye (CRITICAL > HIGH > WARNING > INFO) kazanır

**Alarm ID üretimi:**
```python
alarm_id = f"ALM_{datetime.now():%Y%m%d}_{uuid4().hex[:6].upper()}"
# Örnek: ALM_20240115_A3F9E2
```

**Alarm kaydı** (`fusion engine/api/main.py`):
- `alarms` tablosuna `INSERT OR IGNORE` ile yazılır
- Tablo: edge SQLite `/data/greenhouse.db`

---

### Adım 6 — Alarm Kalıcılık ve Cooldown (Alarm Engine)

| Öğe | Değer |
|-----|-------|
| Kaynak kod | `edge/alarm_engine/engine.py` — `AlarmEngine.process(alarm)` (satır ~55) |
| Çağrılan endpoint | `POST http://edge1_alarm:8107/alarm` |

**Cooldown mekanizması:**
```
_cooldown_tracker: dict[tuple(node_id, rule_id), datetime]

Eğer (node_id, rule_id) için son alarm < ALARM_COOLDOWN_SECONDS (60s) önce tetiklendi:
  → suppressed=True, stored=False (alarm kaydedilmez)
  
Eğer rule_id=None (model monitor kaynaklı):
  → cooldown bypass edilir, her zaman kaydedilir
```

**DB tablosu** (`alarms` tablosuna yazar — alarm engine kendi yazmaz, zaten Fusion Engine yazmış olabilir; ama alarm engine da yazabilir):
```
alarms (
  id INTEGER PRIMARY KEY,
  alarm_id TEXT UNIQUE,
  node_id TEXT,
  timestamp TEXT,
  level TEXT,             -- INFO | WARNING | HIGH | CRITICAL
  source TEXT,            -- rule_engine | model_monitor
  rule_id TEXT,           -- RULE_001 ... RULE_010 veya NULL
  message_key TEXT,
  trigger_values TEXT,    -- JSON
  llm_explanation TEXT,
  synced INTEGER DEFAULT 0   -- 0=unsynced, 1=synced
)
```

**İndeks:** `CREATE INDEX ON alarms(node_id, timestamp)`

---

### Adım 7 — Edge Pipeline Yanıtı

| Öğe | Değer |
|-----|-------|
| Kaynak kod | `edge/gateway/pipeline.py` — `run_inference()` dönüş değeri |
| Yanıt | Alarm objesi (MASTER_SPEC §3.7) veya `{ suppressed: true }` |

Gateway `POST /predict` yanıtı: Alarm objesi veya suppression bilgisi.

---

### Adım 8 — Merkez Senkronizasyon (Pull Protocol)

| Öğe | Değer |
|-----|-------|
| Kaynak kod | `central/aggregator/service.py` — `CentralAggregator.sync_node(node_id)` (satır ~65) |
| Tetikleyen | `start_sync_loop()` — her `CENTRAL_SYNC_INTERVAL` (30s) saniyede |
| Adım 1 | `GET http://edge1_gateway:8100/alerts/unsynced` → `synced=0` alarmlar |
| Adım 2 | Her alarm için `central.db` `alarms` tablosuna `INSERT OR IGNORE` |
| Adım 3 | `POST http://edge1_gateway:8100/alerts/sync-all` → edge alarmları `synced=1` yapılır |
| Adım 4 | `node_registry` tablosu güncellenir: `last_ping`, `last_sync`, `status='ok'` |

**Merkez DB tabloları:**
```
/data/central.db:
  alarms          — edge'den kopyalanan alarmlar (+ llm_explanation eklenir)
  sensor_readings — seed verisi + LSTM için (edge sensör verisi burada değil!)
  node_registry   — kayıtlı node'lar
  monitor_events  — (şu an yazılmıyor — schema'da var ama servis yazmıyor)
  model_outputs   — (şu an yazılmıyor — schema'da var ama servis yazmıyor)
```

**node_registry tablosu:**
```
node_registry (
  node_id TEXT UNIQUE,
  gateway_url TEXT,
  registered_at TEXT,
  last_ping TEXT,
  last_sync TEXT,
  alarm_count INTEGER DEFAULT 0,
  status TEXT DEFAULT 'unknown'
)
```

---

### Adım 9 — LLM Açıklama (Opsiyonel)

| Öğe | Değer |
|-----|-------|
| Kaynak kod | `central/llm_explainer/explainer.py` — `LLMExplainer.explain(alarm)` |
| Tetikleyen | `POST :9000/explain` manuel çağrı (otomatik değil) |

**LLM akışı:**
1. `LLM_API_KEY` env yoksa → direkt template
2. API varsa: `POST https://api.openai.com/v1/chat/completions` (15s timeout)
   ```
   Model: gpt-4o-mini (LLM_MODEL env)
   Max tokens: 200 (LLM_MAX_TOKENS env)
   System: "Sen bir sera yönetim sistemi uzmanısın. Türkçe cevap ver. Maksimum 3 cümle."
   User: "Alarm: {alarm_id}, Seviye: {level}, Kural: {rule_id}, Tetikleyici: {trigger_values}"
   ```
3. Başarısız olursa → `_render_template(alarm)`:
   - `rule_id` → `message_key` eşlemesi
   - `_TEMPLATES` dict'ten Türkçe metin
4. `_update_db_explanation()` → `central.db` `alarms` tablosunda `llm_explanation` güncellenir

**Template mesaj anahtarları:**
| rule_id | message_key | Template (kısa) |
|---------|-------------|---------|
| RULE_001 | fungal_risk_critical | Yüksek nem + hastalık tespiti → fungal enfeksiyon riski kritik |
| RULE_002 | irrigation_required | Toprak nemi düşük, sulama gerekli |
| RULE_003 | sensor_anomaly_high_temp | Sensör anomalisi: aşırı sıcaklık |
| RULE_004 | sensor_anomaly_high_temp | (aynı) |
| RULE_005 | ec_nitrogen_critical | EC düşük, azot eksikliği kritik |
| RULE_006 | ph_nutrition_problem | pH dengesiz, besin alımı bozuk |
| RULE_007 | fungal_risk_critical | (aynı RULE_001) |
| RULE_008 | model_output_drift | Model çıktısı drift → kalibrasyona gönder |
| RULE_009 | fungal_risk_critical | (aynı) |
| RULE_010 | default | Sistem normal çalışıyor |

---

### Adım 10 — Grafana Okuma Yolu

| Grafana | Port | Datasource uid | DB Dosyası | Hangi Tablolar |
|---------|------|----------------|-----------|----------------|
| Central | 3000 | CentralDB | `/data/central.db` | alarms, sensor_readings, node_registry |
| Edge Node 1 | 3001 | EdgeDB | `/data/greenhouse.db` (edge1 volume) | sensor_readings, alarms, monitor_events |
| Edge Node 2 | 3002 | EdgeDB | `/data/greenhouse.db` (edge2 volume) | sensor_readings, alarms, monitor_events |

**Edge Node Dashboard panelleri** (`grafana/dashboards/edge_node.json`):

| Panel | SQL Sorgusu (özet) | Datasource |
|-------|-------------------|------------|
| Sensör Özeti | `SELECT last(temperature), last(humidity), ... FROM sensor_readings WHERE node_id=$node_id` | EdgeDB |
| Sensör Trend | `SELECT timestamp, temperature, humidity, ... FROM sensor_readings WHERE node_id=$node_id ORDER BY timestamp DESC LIMIT 200` | EdgeDB |
| Alarm Geçmişi | `SELECT alarm_id, timestamp, level, rule_id, source FROM alarms WHERE node_id=$node_id ORDER BY timestamp DESC LIMIT 50` | EdgeDB |
| Alarm Seviye Dağılımı | `SELECT level, COUNT(*) as count FROM alarms GROUP BY level` | EdgeDB |
| Alarm Kaynak Dağılımı | `SELECT source, COUNT(*) as count FROM alarms GROUP BY source` | EdgeDB |
| Model Output Monitor | `SELECT timestamp, model_name, metric_value, is_anomaly, z_score FROM monitor_events WHERE node_id=$node_id ORDER BY timestamp DESC LIMIT 50` | EdgeDB |

**Central Dashboard panelleri** (`grafana/dashboards/central.json`):

| Panel | SQL Sorgusu (özet) | Datasource |
|-------|-------------------|------------|
| Node Karşılaştırma | `SELECT node_id, COUNT(*) as alarms FROM alarms GROUP BY node_id` | CentralDB |
| Birleşik Alarm Akışı | `SELECT alarm_id, node_id, level, rule_id, llm_explanation FROM alarms ORDER BY timestamp DESC LIMIT 100` | CentralDB |
| Alarm Trend | `SELECT date(timestamp) as day, COUNT(*) as count FROM alarms GROUP BY day` | CentralDB |
| Kaynak Dağılımı | `SELECT source, COUNT(*) FROM alarms GROUP BY source` | CentralDB |
| LLM Açıklamaları | `SELECT alarm_id, level, llm_explanation FROM alarms WHERE llm_explanation IS NOT NULL ORDER BY timestamp DESC` | CentralDB |
| Node Durumu | `SELECT node_id, status, last_sync, alarm_count FROM node_registry` | CentralDB |

**Grafana → SQLite okuma yolu:**
```
Grafana HTTP query → frser-sqlite-datasource plugin → SQLite Go driver
→ /data/greenhouse.db (ya da central.db) → doğrudan dosya okuma (HTTP yok)
```

Datasource, container'a bağlı Docker volume üzerindeki SQLite dosyasına doğrudan erişir. Grafana'nın yeniden başlatılması buffer kaybına yol açmaz; veritabanı volume'de kalır.

---

## Hata Yolları

### Model Servisi Çöktüğünde

```
EdgePipeline.run_inference()
  asyncio.gather(return_exceptions=True) çalışır
  → Exception döner (Exception örneği, dict değil)
  
EdgePipeline._process_model_result(result, model_name)
  → isinstance(result, Exception) → True
  → _get_fallback(model_name, node_id, timestamp) çağrılır
  → Nötr değer eklenir
  
Pipeline devam eder:
  → Fusion Engine nötr değerlerle çalışır
  → Kural eşiklerini karşılamayan değerler → INFO alarm (RULE_010)
```

**Etki**: Tek model çökmesi tüm pipeline'ı durdurmaz. `confidence=0.0` değerleri RULE_001 gibi yüksek güven gerektiren kuralları engelleyebilir.

---

### Edge → Merkez Bağlantı Koptuğunda

```
CentralAggregator.sync_node(node_id):
  GET /alerts/unsynced → httpx.ConnectError
  → except Exception as e: → log yazılır, sync_status="error"
  → node_registry güncellenmez (last_sync değişmez)
  
Edge alarmları "synced=0" olarak kalır
Edge 8107'de `alarms` tablosu büyümeye devam eder
Edge tamamen bağımsız çalışır (8100-8107 kendi içinde bütün)

Merkez geri geldiğinde:
  sync_node() → GET /alerts/unsynced → birikmiş alarmlar
  → INSERT OR IGNORE (alarm_id UNIQUE kısıtı ile çakışma yok)
  → POST /alerts/sync-all
  → 0 veri kaybı
```

**Kritik garanti**: `INSERT OR IGNORE alarm_id UNIQUE` — aynı alarm iki kez senkronize edilse bile merkez DB'de tek kez görünür.

---

### LLM API Başarısız Olduğunda

```
LLMExplainer.explain(alarm):
  requests.post(OPENAI_URL, ..., timeout=15) → exception (timeout/network/auth)
  → except Exception: → _render_template(alarm) çağrılır
  
_render_template(alarm):
  alarm.rule_id → _RULE_TO_MESSAGE_KEY dict
  → _TEMPLATES[message_key] → Türkçe sabit metin
  → source="template" olarak dönülür
```

**Etki**: Açıklama kalitesi düşer ama servis 503 dönmez. `source` alanı `"llm"` veya `"template"` olarak kayıt altında.

---

### Model Eğitimi Başarısız Olduğunda (Docker Compose Başlangıcı)

```
docker-compose.yml: training servisleri (restart: "no")
Eğer F1/accuracy hedefi tutturulmazsa:
  train.py → ValueError → container exit code 1
  
İnference servisleri (disease/irrigation/nutrition/anomaly):
  depends_on: {train_service: {condition: service_completed_successfully}}
  → Eğitim başarısız olursa inference servisi başlamaz
  
edge_gateway:
  depends_on: {edge1_disease: {condition: service_healthy}} → sağlıklı olana kadar bekler
  → Tüm eğitimler bitene kadar pipeline başlamaz
```

---

### Yeni Node Ekleme (Hata Riski)

```
Yeni node "greenhouse_03" eklense:
1. central/aggregator/api/main.py:
   NODE1_GATEWAY_URL, NODE2_GATEWAY_URL → NODE3_GATEWAY_URL eklenmesi gerekir
   (startup'ta sadece 2 node kaydediliyor)
   
2. central/lstm_yield/api/main.py:
   KNOWN_NODES = ["greenhouse_01", "greenhouse_02"]
   → "greenhouse_03" GET /yield'de görünmez
   
3. central/lstm_yield/train.py:
   node_ids = ["greenhouse_01", "greenhouse_02"]
   → Yeni node'un sensör verisi LSTM eğitimine dahil edilmez
   
Şu an için manuel kayıt:
   POST http://localhost:9001/nodes/register {"node_id": "greenhouse_03", "gateway_url": "..."}
   Bu yalnızca aggregator sync için yeterli; LSTM için kod değişikliği şart.
```

---

## Tam Veri Akışı Özeti (Tablo)

| Adım | İşlem | Kaynak Fonksiyon | DB Tablosu | Dosya |
|------|-------|------------------|------------|-------|
| 1 | Sensör üretimi | `SensorSimulator.generate_reading()` | `sensor_readings` → edge DB | `simulator/sensor_simulator.py` |
| 2a | Disease tahmin | `DiseaseInference.predict()` | — | `models/disease/inference.py` |
| 2b | Irrigation tahmin | `IrrigationInference.predict()` | — | `models/irrigation/inference.py` |
| 2c | Nutrition tahmin | `NutritionInference.predict()` | — | `models/nutrition/inference.py` |
| 2d | Anomaly tahmin | `AnomalyInference.predict()` | — | `models/anomaly/inference.py` |
| 3 | Monitor kontrolü | `MonitorService.check()` | `monitor_events` → edge DB | `models/output_monitor/monitor.py` |
| 4 | Fusion input oluştur | `EdgePipeline._build_fusion_input()` | — | `edge/gateway/pipeline.py` |
| 5 | Kural değerlendirme | `RuleEngine.evaluate()` → `FusionEngine.process()` | `alarms` → edge DB | `edge/fusion_engine/engine.py` |
| 6 | Alarm cooldown | `AlarmEngine.process()` | `alarms` → edge DB | `edge/alarm_engine/engine.py` |
| 7 | Pipeline yanıtı | `EdgePipeline.run_inference()` dönüş | — | `edge/gateway/pipeline.py` |
| 8a | Merkez pull | `CentralAggregator.sync_node()` | — | `central/aggregator/service.py` |
| 8b | Merkez yazma | `CentralAggregator._write_alarms()` | `alarms` → central DB | `central/aggregator/service.py` |
| 8c | Edge işaretle | `POST /alerts/sync-all` | `alarms.synced=1` → edge DB | `edge/alarm_engine/api/main.py` |
| 9 | LLM açıklama | `LLMExplainer.explain()` | `alarms.llm_explanation` → central DB | `central/llm_explainer/explainer.py` |
| 10 | Grafana okuma | frser-sqlite-datasource | `alarms`, `sensor_readings` | `monitoring/grafana/dashboards/` |

---

## DB Schema Referansı

### edge SQLite: `/data/greenhouse.db`

Tablo | Yazan Servis(ler) | Okuyan Servis(ler)
--- | --- | ---
`sensor_readings` | `sensor_simulator.py` (SensorSimulator._write_db) | edge_gateway (GET /logs), edge1_grafana
`model_outputs` | **Hiçbiri** (schema'da var, kod yok) | —
`alarms` | fusion_engine/api, alarm_engine/engine | alarm_engine/api (GET /alerts, GET /unsynced), central_aggregator (pull), edge1_grafana
`monitor_events` | output_monitor/api (DB_PATH varsa) | edge1_grafana
`node_registry` | — (sadece central DB'de kullanılıyor) | —

### central SQLite: `/data/central.db`

Tablo | Yazan Servis(ler) | Okuyan Servis(ler)
--- | --- | ---
`alarms` | central_aggregator/service (sync), central_llm/api (explain) | central_aggregator/api (GET /alerts), central_grafana
`sensor_readings` | `scripts/seed_historical_data.py` (seed) | central_lstm/inference (LSTM girdisi), central_grafana
`node_registry` | central_aggregator/service (register + sync) | central_aggregator/api (GET /nodes), central_grafana
`model_outputs` | **Hiçbiri** | —
`monitor_events` | **Hiçbiri** | —
