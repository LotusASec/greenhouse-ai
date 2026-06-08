# Greenhouse AI — Codebase Map

## Nasıl Okunur
Bu doküman projedeki her Python, YAML, JSON ve shell dosyasını satır satır okuarak üretilmiştir; teorik değil, gerçek kod davranışını yansıtır. Bir servis ya da fonksiyon hakkında soru çıktığında burası başlangıç noktasıdır.

## Sistem Mimarisi

```
┌──────────────────────────────────────────────────────────────────┐
│  EDGE NODE 1 (greenhouse_01)                                     │
│                                                                  │
│  sensor_simulator ──► edge1_gateway (8100)                       │
│  image_simulator  ──►   ├─► edge1_disease      (8101) ─┐        │
│                         ├─► edge1_irrigation   (8102) ─┤        │
│                         ├─► edge1_nutrition    (8103) ─┤        │
│                         ├─► edge1_anomaly      (8104) ─┤        │
│                         ├─► edge1_output_monitor(8105)─┘        │
│                         ├─► edge1_fusion       (8106)           │
│                         └─► edge1_alarm        (8107)           │
│                                                                  │
│  SQLite: /data/greenhouse.db (edge1_db volume)                   │
│  Grafana: edge1_grafana (3001) ◄── EdgeDB datasource            │
└──────────────────────────────┬───────────────────────────────────┘
                               │  HTTP /alerts/unsynced + /alerts/sync-all
                               │  (pull, every 30s)
┌──────────────────────────────▼───────────────────────────────────┐
│  CENTRAL LAYER                                                   │
│                                                                  │
│  central_gateway (9000)                                          │
│    ├─► central_aggregator (9001) ──► central.db                  │
│    ├─► central_lstm       (9002) ──► central.db sensor_readings  │
│    └─► central_llm        (9003) ──► central.db alarms           │
│                                                                  │
│  SQLite: /data/central.db (central_db volume)                    │
│  Grafana: grafana (3000) ◄── CentralDB datasource               │
└──────────────────────────────────────────────────────────────────┘
```

*(Edge Node 2 greenhouse_02 port aralığı: 8200-8207, 3002)*

---

## Dizin Yapısı

---

### simulator/

**sensor_simulator.py**
- Ne yapar: YAML profil dosyasından node konfigürasyonu okur; güneş döngüsü (sinüsoidal), Gaussian gürültü ve anomali enjeksiyonu ile gerçekçi sensör okumaları üretir.
- Girdi: `config/node1.yaml` veya `node2.yaml` (argüman); `DB_PATH` ortam değişkeni
- Çıktı: `sensor_readings` tablosuna SQLite yazısı; `dict` döner (`generate_reading()`)
- Kritik fonksiyonlar:
  - `SensorSimulator.__init__` → config yükler, DB bağlantısı kurar
  - `SensorSimulator.generate_reading()` → MASTER_SPEC §3.1 formatında tek okuma üretir
  - `SensorSimulator.inject_anomaly(profile)` → `spike` / `freeze` / `drift` modları
  - `SensorSimulator.run()` → sonsuz döngü, Ctrl-C ile durur
- Bağımlılıklar: SQLite (`DB_PATH`), PyYAML; HTTP bağlantısı yok

**image_simulator.py**
- Ne yapar: Node YAML'ındaki `image_class_distribution` ağırlıklarına göre PlantVillage görüntülerini örnekler; base64 kodlar.
- Girdi: `config/node1.yaml`, dataset dizini (`DATASET_PATH`), `models/disease/config/column_map.yaml`
- Çıktı: `{ image_b64, ground_truth_class, node_id, timestamp }` dict listesi
- Kritik fonksiyonlar:
  - `ImageSimulator.__init__` → index oluşturur, sınıf dağılımını doğrular
  - `ImageSimulator.sample(n)` → n adet örnek döner
  - `ImageSimulator._encode_image(path)` → 224×224 JPEG → base64
- Bağımlılıklar: Pillow (opsiyonel — yoksa ham bytes kullanır), PyYAML; HTTP yok

**config/node1.yaml**
- `node_id: greenhouse_01`, `profile: node1`
- Sera A – Domates; sıcaklık ortalaması 24°C
- `image_class_distribution`: %80 healthy, %10 early_blight, ...
- `emit_interval_seconds: 1`
- `anomaly_injection.enabled: false`

**config/node2.yaml**
- `node_id: greenhouse_02`, `profile: node2`
- Sera B – Salatalık; sıcaklık ortalaması 23°C
- `image_class_distribution`: %92 healthy
- Anomali enjeksiyonu kapalı

---

### models/disease/

**train.py**
- Ne yapar: ResNet-50 (ImageNet pretrained) ile transfer learning; son 2 katman ince ayarı.
- Girdi: `data/train`, `data/val`, `data/test` (ImageFolder); `config/column_map.yaml` sınıf isimleri
- Çıktı: `model_weights/disease_model.pth` (state_dict + class_names + metrikler)
- Kritik fonksiyonlar:
  - `build_model(num_classes)` → layer4 dondurulur, fc değiştirilir
  - `remap()` → folder sırası ile config sırası hizalanır
  - `evaluate_f1()` → weighted F1 hesaplar; hedef ≥0.85
- Bağımlılıklar: PyTorch, torchvision, scikit-learn; dosya I/O

**inference.py**
- Ne yapar: `disease_model.pth` checkpoint'ini yükler; base64 görüntü → sınıf olasılıkları.
- Girdi: `{ node_id, timestamp, image_b64 }` dict
- Çıktı: MASTER_SPEC §3.2 formatında dict
- Kritik fonksiyonlar:
  - `DiseaseInference.__init__` → checkpoint'ten class_names yükler (hardcode yok)
  - `DiseaseInference.predict(input_data)` → val_transform uygular, softmax, top_prediction
- Bağımlılıklar: PyTorch, Pillow; HTTP yok

**api/main.py**
- Ne yapar: Sadece HTTP katmanı; `DiseaseInference` başlatır ve POST /predict endpoint'i sunar.
- Port: `SERVICE_PORT` env (varsayılan 8101)
- Endpoint'ler: `GET /health`, `POST /predict`
- Model yüklenemezse 503 döner (servis devam eder)

**scripts/download_dataset.py**
- Ne yapar: HuggingFace veya lokal dizinden PlantVillage görüntülerini indirir ve `train/val/test` klasör yapısına organize eder (70/15/15 split).
- Girdi: `column_map.yaml` sınıf eşlemesi
- Çıktı: `data/train/healthy/`, `data/train/early_blight/`, vb.

**config/column_map.yaml**
- `class_mapping`: `healthy → Tomato___healthy`, `early_blight → Tomato___Early_blight`, ...
- Sağ tarafı değiştirmek yeterli; Python kodu değişmez.

---

### models/irrigation/

**train.py**
- Ne yapar: XGBoostClassifier (irrigate: bool) + XGBoostRegressor (amount_liters) eğitir.
- Girdi: `../shared/data/irrigation_train.csv`, `irrigation_test.csv`; `config/column_map.yaml`
- Çıktı: `model_weights/irrigation_classifier.joblib`, `irrigation_regressor.joblib`
- Kritik fonksiyonlar:
  - `get_feature_cols(col_map)` → TR-11 özellik sırası
  - `train_classifier`, `train_regressor` → XGBoost modelleri
  - Doğruluk kapısı: accuracy < 0.90 → ValueError
- Bağımlılıklar: xgboost, scikit-learn, pandas, joblib

**inference.py**
- Ne yapar: İki joblib modeli yükler; sensör dict → `{ irrigate, amount_liters, confidence }`.
- Kritik: `amount_liters` 0-10 L arasına kesilir; irrigate=False ise amount=0.0
- Bağımlılıklar: xgboost, joblib, numpy, yaml

**api/main.py**
- Port: 8102
- Endpoint'ler: `GET /health`, `POST /predict`

---

### models/nutrition/

**train.py**
- Ne yapar: RandomForestClassifier 4-sınıf (N/P/K eksikliği, normal) eğitir.
- Girdi: `nutrition_train.csv`, `nutrition_test.csv`; `config/column_map.yaml`
- Çıktı: `model_weights/nutrition_model.joblib`
- Doğruluk kapısı: weighted F1 < 0.88 → ValueError
- Bağımlılıklar: scikit-learn, pandas, joblib

**inference.py**
- Ne yapar: RF modeli yükler; sensör dict → `{ deficiency_class, fertilizer_recommendation, confidence }`.
- Kritik: `FERTILIZER_RECOMMENDATIONS` dict — hardcode görünüyor ama spec gereği (TR-11 değil, öneri metinleri)
- Bağımlılıklar: scikit-learn, joblib, numpy

**api/main.py**
- Port: 8103
- Endpoint'ler: `GET /health`, `POST /predict`

---

### models/anomaly/

**train.py**
- Ne yapar: IsolationForest (contamination=0.05) sadece normal verilerle eğitilir.
- Girdi: `anomaly_normal.csv`, `anomaly_test_normal.csv`, `anomaly_test_inject.csv`
- Çıktı: `model_weights/anomaly_model.joblib`
- Doğruluk kapısı: precision < 0.88 → ValueError
- Bağımlılıklar: scikit-learn, pandas, joblib

**inference.py**
- Ne yapar: IF modeli yükler; sensör dict → `{ is_anomaly, anomaly_score }`.
- Kritik: `anomaly_score = sigmoid(raw_score × 5)` — raw_score < 0 ise yüksek skor, raw > 0 ise düşük
- Bağımlılıklar: scikit-learn, joblib, numpy

**api/main.py**
- Port: 8104
- Endpoint'ler: `GET /health`, `POST /predict`

---

### models/output_monitor/

**monitor.py**
- Ne yapar: 4 model için (disease, irrigation, nutrition, anomaly) ayrı sliding-window buffer tutar; z-score + IQR ile anomali tespiti yapar.
- Girdi: `model_name`, `metric_value`, `node_id`, `timestamp`
- Çıktı: MASTER_SPEC §3.6 formatında `MonitorEvent` dict
- Kritik parametreler: `WINDOW_SIZE=50`, `MIN_SAMPLES=10`, `Z_THRESHOLD=2.5`
- Kritik fonksiyonlar:
  - `MonitorService.check()` → z-score ve IQR ikisi de anomali işaretlerse `is_anomaly=True`
  - `MonitorService._model_status()` → buffer istatistikleri
  - `MonitorService.reset()` → testlerde buffer sıfırlama
- **Önemli**: Buffer sadece bellekte; servis yeniden başladığında sıfırlanır.

**api/main.py**
- Port: 8105
- Endpoint'ler:
  - `GET /health`
  - `POST /monitor/{model_name}` → tek model için kontrol, `model_name` ∈ {disease, irrigation, nutrition, anomaly}
  - `GET /status` → tüm model buffer durumları
  - `GET /status/{model_name}` → tek model durumu
- DB: `DB_PATH` varsa `monitor_events` tablosuna yazar

---

### models/shared/

**generate_training_data.py**
- Ne yapar: irrigation, nutrition ve anomaly modelleri için sentetik CSV eğitim verisi üretir (kural tabanlı etiketler).
- Girdi: Yok (parametreler sabit/argüman)
- Çıktı: `data/` klasöründe 7 CSV dosyası
- Kritik: Anomali tipler multi-feature extremes (5 tür × 5 satır = 25 satır inject)

---

### edge/fusion_engine/

**engine.py**
- Ne yapar: `RuleEngine` (rules.yaml yükler, field path değerlendirme), `FusionEngine` (kural motoru + monitor olaylarını birleştirir, alarm objesi oluşturur).
- Girdi: `FusionInput` dict (MASTER_SPEC §3.8)
- Çıktı: Alarm dict (MASTER_SPEC §3.7)
- Kritik fonksiyonlar:
  - `RuleEngine.evaluate(fusion_input)` → ilk eşleşen kuralı döner; eşleşme yoksa INFO/no_rule_matched
  - `RuleEngine._get_nested(data, field_path)` → noktalı path ile dict gezinme
  - `RuleEngine.update_threshold(rule_id, field, value)` → kuralı günceller ve `rules.yaml`'a geri yazar
  - `FusionEngine._merge_results(rule, monitor)` → yüksek seviyeli kazanır
  - `FusionEngine._build_alarm()` → `ALM_YYYYMMDD_XXXXXX` formatında ID üretir
- Bağımlılıklar: PyYAML; HTTP yok

**api/main.py**
- Port: 8106
- Endpoint'ler:
  - `GET /health` → yüklenen kural sayısını da döner
  - `POST /evaluate` → FusionInput alır, Alarm döner, DB'ye yazar
  - `GET /rules` → 10 kuralın listesi
  - `PUT /rules/{rule_id}/threshold` → kural eşiği güncelleme
- DB: `DB_PATH=/data/greenhouse.db`, `alarms` tablosuna yazar

**rules/rules.yaml**
- 10 kural: RULE_001 (CRITICAL fungal) ... RULE_010 (INFO all normal)
- Yapı: `id, name, conditions[], action.alarm_level, action.message_key`
- `conditions[].field`: noktalı path (`sensor_reading.sensors.humidity`)
- `conditions[].operator`: `gt | lt | eq | neq | gte | lte`
- **ÖNEMLİ**: Bu dosya `update_threshold()` tarafından değiştirilebilir; container yeniden başlatıldığında kaybolur.

---

### edge/alarm_engine/

**engine.py**
- Ne yapar: Alarm kalıcılığı, cooldown tabanlı tekrar bastırma, geçmiş sorgulama ve senkronizasyon durumu yönetimi.
- Girdi: Alarm dict (MASTER_SPEC §3.7)
- Çıktı: `{ stored: bool, suppressed: bool, alarm: dict }`
- Kritik fonksiyonlar:
  - `AlarmEngine.process(alarm)` → cooldown kontrolü → SQLite'a yazar
  - `AlarmEngine._is_suppressed(alarm)` → (node_id, rule_id) çiftini izler; `rule_id=None` (monitor) her zaman geçer
  - `AlarmEngine.get_alerts(level, source, since, limit, offset)` → filtrelenmiş liste
  - `AlarmEngine.get_unsynced()` → `synced=0` olan alarmlar
  - `AlarmEngine.mark_all_synced()` → merkez tarafından çekildikten sonra çağrılır
- Bağımlılıklar: SQLite (`DB_PATH`); HTTP yok
- Varsayılan cooldown: 60 saniye (`ALARM_COOLDOWN_SECONDS` env)

**api/main.py**
- Port: 8107
- Endpoint'ler:
  - `GET /health`
  - `POST /alarm` → alarm depola
  - `GET /alerts` → filtrelenmiş geçmiş
  - `GET /alerts/{alarm_id}` → tek alarm
  - `GET /alerts/summary` → level/source bazlı sayım
  - `GET /alerts/unsynced` → `synced=0` alarmlar
  - `POST /alerts/{alarm_id}/sync` → tek alarm işaretle
  - `POST /alerts/sync-all` → tüm unsynced → synced

---

### edge/gateway/

**pipeline.py**
- Ne yapar: Tüm edge servislerini HTTP üzerinden orkestre eder; sensör okuma → alarm döngüsü.
- Kritik fonksiyonlar:
  - `EdgePipeline.__init__` → SensorSimulator + ImageSimulator başlatır, URL'leri env'den okur
  - `EdgePipeline.run_inference(sensor_reading, image_b64)` → 4 model paralel çağrısı → monitor → fusion → alarm
  - `EdgePipeline.start_continuous_loop()` → `emit_interval` saniyede bir döngü
  - `EdgePipeline.check_all_services()` → 7 servise `/health` ping
  - `EdgePipeline.get_logs(limit, offset)` → SQLite `sensor_readings` okur
  - `EdgePipeline._get_fallback(model_name, node_id, timestamp)` → model çökerse nötr değer
- Fallback davranışı: Herhangi bir model başarısız olursa nötr değerle devam eder, pipeline durmaz.
- **Kritik not**: `image_b64=None` olursa disease modeli fallback değer döner (DATASET_PATH ayarsızsa bu olur).

**main.py**
- Ne yapar: FastAPI uygulaması; lifespan'da EdgePipeline başlatır ve continuous loop'u background task olarak çalıştırır.
- Port: 8100
- Endpoint'ler:
  - `GET /health` → tüm 7 downstream servis durumu
  - `POST /predict` → tek tam pipeline çalıştırır
  - `GET /status` → loop istatistikleri + servis durumları
  - `GET /threshold` → Fusion Engine `/rules` proxy
  - `PUT /threshold/{rule_id}` → Fusion Engine `/rules/{rule_id}/threshold` proxy
  - `GET /logs` → SQLite sensor_readings proxy
  - `GET /alerts` → Alarm Engine `/alerts` proxy
  - `GET /sync` → Alarm Engine `/alerts/unsynced` proxy
  - `GET /alerts/unsynced` → `/sync` ile aynı (merkez için alias)
  - `POST /alerts/sync-all` → Alarm Engine `/alerts/sync-all` proxy

---

### central/aggregator/

**service.py**
- Ne yapar: Her edge node'dan periyodik pull sync; merkez SQLite'a yazar; node kayıt tutma.
- Girdi: Node URL'leri (env), alarm dict listesi (edge'den çekilen)
- Çıktı: Merkez SQLite `alarms`, `node_registry` tabloları
- Kritik fonksiyonlar:
  - `CentralAggregator.register_node(node_id, gateway_url)` → hem bellek hem DB'ye yazar
  - `CentralAggregator.sync_node(node_id)` → GET unsynced → DB'ye yaz → POST sync-all
  - `CentralAggregator.start_sync_loop()` → `CENTRAL_SYNC_INTERVAL` saniyede bir
  - `CentralAggregator.get_all_alerts(filters)` → merkez DB'den filtrelenmiş liste
  - `CentralAggregator.update_alarm_explanation(alarm_id, explanation)` → LLM açıklamasını günceller
- Bağımlılıklar: httpx, SQLite

**aggregator.py**
- Ne yapar: Faz 1 stub; `aggregate_nodes()` `None` döner. Gerçek mantık `service.py`'de.

**api/main.py**
- Port: 9001
- Lifespan: greenhouse_01 ve greenhouse_02 node'larını kaydeder, sync loop başlatır
- Endpoint'ler:
  - `GET /health`
  - `GET /nodes` → kayıtlı node listesi
  - `GET /nodes/{node_id}/status`
  - `GET /nodes/{node_id}/alerts`
  - `POST /nodes/register`
  - `POST /nodes/{node_id}/sync` → tek node elle senkronize
  - `GET /alerts` → filtrelenmiş birleşik alarmlar
  - `GET /alerts/summary`
  - `GET /status` → sync durumu
  - `POST /alarms/{alarm_id}/explain` → LLM açıklamasını DB'ye geri yazar

---

### central/lstm_yield/

**lstm.py**
- Ne yapar: Faz 1 stub; `forecast()` `None` döner. Gerçek LSTM `train.py`'de.

**train.py**
- Ne yapar: Merkez DB `sensor_readings`'ten günlük ortalama alır; 7 günlük pencereler oluşturur; 2-layer LSTM eğitir.
- Girdi: `/data/central.db` (CENTRAL_DB_PATH)
- Çıktı: `model_weights/yield_model.pth`
- Kritik: En az 14 gün veri gerekli; synthetic verim skoru (temp + hum + ec uyumu)
- Node'lar: `greenhouse_01` ve `greenhouse_02` birleştirilir.

**inference.py**
- Ne yapar: `yield_model.pth` yükler; merkez DB'den son 7 günlük ortalama alır → normalize → tahmin.
- Çıktı: `{ node_id, status, yield_score (0-1), confidence: low/medium/high, days_used }`
- Veri yoksa: `{ status: insufficient_data, min_days_required: 14, days_available: N }`
- Model yoksa: `{ status: model_not_trained }`

**api/main.py**
- Port: 9002
- Endpoint'ler:
  - `GET /health` → `model_ready: bool` (pth dosyası var mı)
  - `GET /yield/{node_id}`
  - `GET /yield` → tüm node'lar (hardcode: greenhouse_01, greenhouse_02)

**scripts/seed_historical_data.py**
- Ne yapar: Merkez DB'ye 21 gün × 24 saat × 2 node = 1008 satır sentetik sensör verisi ekler (LSTM eğitimi için).
- `start_demo.sh` tarafından otomatik çağrılır.

---

### central/llm_explainer/

**explainer.py**
- Ne yapar: Alarm objesi → Türkçe açıklama. Önce OpenAI API dener; başarısız olursa template kullanır.
- Kritik: `LLM_API_KEY` env yoksa direkt template fallback
- Template map: RULE_001 → fungal_risk_critical, RULE_002 → irrigation_required, vb.
- OpenAI endpoint: `https://api.openai.com/v1/chat/completions`; 15 saniye timeout

**api/main.py**
- Port: 9003
- Endpoint'ler:
  - `GET /health` → `llm_available: bool | null`
  - `POST /explain` → AlarmObject → ExplainResponse
- POST /explain sonrası merkez DB `alarms` tablosu güncellenir (`llm_explanation` alanı).

---

### central/gateway/

**api/main.py**
- Ne yapar: Tüm merkez servislerini proxy eder; kendi iş mantığı yok.
- Port: 9000
- Proxy hedefleri: AGG_URL (:9001), LSTM_URL (:9002), LLM_URL (:9003)
- Endpoint'ler:
  - `GET /health` → 3 servis durumu
  - `GET /status` → aggregator'dan proxy
  - `GET /nodes`, `/nodes/{id}/status`, `/nodes/{id}/alerts`, `POST /nodes/register`
  - `GET /alerts`, `/alerts/summary`
  - `GET /yield`, `/yield/{node_id}`
  - `POST /explain`

---

### database/

**schema.sql**
- 5 tablo, hepsi `CREATE TABLE IF NOT EXISTS` (idempotent):
  - `sensor_readings` — node_id, timestamp, 6 sensör değeri
  - `model_outputs` — node_id, model_name, top_prediction, confidence, class_probabilities (JSON), raw_output (JSON)
  - `alarms` — alarm_id (UNIQUE), node_id, level, source, rule_id, trigger_values (JSON), llm_explanation, synced (0/1)
  - `monitor_events` — node_id, model_name, metric_value, z_score, window_mean, window_std, is_anomaly
  - `node_registry` — node_id (UNIQUE), gateway_url, last_ping, last_sync, status
- Her tabloda `(node_id, timestamp)` kompozit index
- **Not**: `model_outputs` tablosu schema'da var ama **hiçbir servis bu tabloya yazmıyor** (edge gateway pipeline, sadece alarms ve sensor_readings kullanıyor).

**init.py**
- `DB_PATH` env değişkeninden yolu okur; `schema.sql` çalıştırır.
- Idempotent; birden fazla kez çalıştırılabilir.

---

### monitoring/

**Dockerfile**
- `FROM grafana/grafana:10.4.0`
- `grafana-cli plugins install frser-sqlite-datasource`
- `provisioning/` → `/etc/grafana/provisioning/`
- `dashboards/` → `/var/lib/grafana/dashboards/`
- Admin: `admin` / `greenhouse`
- Varsayılan ev dashboard: `edge_node.json` (compose'da central için `central.json` override edilir)

**grafana/provisioning/datasources/edge_datasource.yaml**
- name: `EdgeDB`, uid: `EdgeDB`
- type: `frser-sqlite-datasource`
- path: `/data/greenhouse.db`
- `isDefault: false`

**grafana/provisioning/datasources/central_datasource.yaml**
- name: `CentralDB`, uid: `CentralDB`
- type: `frser-sqlite-datasource`
- path: `/data/central.db`
- `isDefault: true`

**grafana/provisioning/dashboards/dashboard_provider.yaml**
- Dashboard klasörü: `/var/lib/grafana/dashboards`
- `updateIntervalSeconds: 30`; `disableDeletion: true`

**grafana/dashboards/edge_node.json**
- 6 panel: Sensör Özeti (stat), Sensör Trend (zaman serisi), Alarm Geçmişi (tablo), Alarm Seviye Dağılımı (pie), Alarm Kaynak Dağılımı (bar), Model Output Monitor (tablo)
- Datasource: `EdgeDB` (uid)
- Yenileme: 5 saniye; varsayılan zaman: son 24 saat
- `$node_id` değişkeni ile node filtreleme; varsayılan `greenhouse_01`

**grafana/dashboards/central.json**
- 6 panel: Node Karşılaştırma (stat), Birleşik Alarm Akışı (tablo), Alarm Trend (zaman serisi), Kaynak Dağılımı (pie), LLM Açıklamaları (tablo), Node Durumu (tablo — node_registry)
- Datasource: `CentralDB` (uid)
- Yenileme: 10 saniye

---

### scripts/

**start_demo.sh**
- `docker compose up -d` → `wait_for_services.sh` → `seed_historical_data.py` → `health_check.sh`
- Başarılı çıkışta URL'leri listeler

**wait_for_services.sh**
- 12 servis için (edge + central + 3 Grafana) maksimum 600 saniye bekler, 5 saniye aralıklı polling
- Grafana için `/api/health`, diğerleri için `/health` endpoint'i kullanır

**health_check.sh**
- Aynı 15 servis için tek seferlik kontrol; tablo formatında çıktı; başarısız varsa exit 1

---

### tests/

**unit/test_fusion_engine.py**
- 10 test: Her kural (RULE_001/002), monitor drift, kural vs monitor kazanma, alarm_id formatı, nested field, geçersiz operator, YAML doğrulama
- Doğrudan dosya yolu ile import eder (modül adı çakışması önler)

**unit/test_alarm_engine.py**
- 10 test: Depolama, cooldown bastırma, cooldown süresi, monitor bypass, level/since filtre, özet sayım, sync işaretleme, sayfalama, bilinmeyen ID
- Mock datetime kullanır (sleep yok)

**unit/test_monitor.py**
- 6 test: Isınma periyodu, normal değerler, spike tespiti, freeze koruması, bağımsız buffer, buffer rollover

**integration/test_edge_pipeline.py** — Edge gateway 8100'e karşı 8 test
**integration/test_central_layer.py** — Merkez 9000'e karşı 8 test
**integration/test_fusion_engine.py** — Fusion 8106'ya karşı testler
**integration/test_alarm_engine.py** — Alarm 8107'ye karşı testler
**integration/test_monitor_service.py** — Monitor 8105'e karşı testler
**integration/test_disease_model.py** — Disease 8101'e karşı testler
**integration/test_sensor_models.py** — Irrigation/Nutrition/Anomaly modelleri
**integration/test_full_system.py** — Uçtan uca sistem testi

**demo/inject_scenario.py**
- 5 senaryo: `normal` (INFO), `fungal` (CRITICAL), `dry` (WARNING), `anomaly` (HIGH), `drift` (WARNING)
- POST /predict ile gerçek pipeline'ı tetikler
- Drift senaryosu direkt monitor servisine 51 değer gönderir

**demo/measure_kpi.py**
- 4 KPI ölçümü: alarm latency, LLM latency, monitor precision, anomaly precision
- `KPI_REPORT.md` üretir

**demo/test_disconnection.py**
- Merkez servislerini durdurur → edge üretir → merkez başlatır → sync doğrular → 0 veri kaybı doğrular
- `docker compose stop/start` kullanır

**smoke/test_grafana.sh**
- 3 Grafana instance'ına /api/health; CentralDB datasource var mı; ≥2 dashboard yüklendi mi; SQLite sorgu çalışıyor mu

---

## Servis Port Haritası

| Servis | Host Port | Container Port | Görev | DB Bağlantısı |
|--------|-----------|----------------|-------|----------------|
| edge1_gateway | 8100 | 8100 | Edge pipeline orkestrasyonu | sensor_readings okur |
| edge1_disease | 8101 | 8101 | ResNet-50 hastalık tespiti | Yok |
| edge1_irrigation | 8102 | 8102 | XGBoost sulama kararı | Yok |
| edge1_nutrition | 8103 | 8103 | RandomForest besin eksikliği | Yok |
| edge1_anomaly | 8104 | 8104 | IsolationForest anomali | Yok |
| edge1_output_monitor | 8105 | 8105 | Z-score model izleme | monitor_events yazar |
| edge1_fusion | 8106 | 8106 | Kural motoru + alarm üretimi | alarms yazar |
| edge1_alarm | 8107 | 8107 | Alarm kalıcılık + cooldown | alarms okur/yazar |
| edge2_gateway | 8200 | 8100 | Edge 2 pipeline (node2) | sensor_readings okur |
| edge2_disease | 8201 | 8101 | Node 2 hastalık modeli | Yok |
| edge2_irrigation | 8202 | 8102 | Node 2 sulama | Yok |
| edge2_nutrition | 8203 | 8103 | Node 2 besin | Yok |
| edge2_anomaly | 8204 | 8104 | Node 2 anomali | Yok |
| edge2_output_monitor | 8205 | 8105 | Node 2 monitor | monitor_events yazar |
| edge2_fusion | 8206 | 8106 | Node 2 fusion | alarms yazar |
| edge2_alarm | 8207 | 8107 | Node 2 alarm | alarms okur/yazar |
| central_gateway | 9000 | 9000 | Merkez proxy / giriş noktası | Yok |
| central_aggregator | 9001 | 9001 | Edge→Merkez sync + sorgu | alarms, node_registry yazar |
| central_lstm | 9002 | 9002 | LSTM verim tahmini | sensor_readings okur |
| central_llm | 9003 | 9003 | LLM açıklama üretimi | alarms günceller |
| grafana (central) | 3000 | 3000 | Merkez Grafana dashboard | central.db okur |
| edge1_grafana | 3001 | 3000 | Edge 1 Grafana dashboard | greenhouse.db (edge1) okur |
| edge2_grafana | 3002 | 3000 | Edge 2 Grafana dashboard | greenhouse.db (edge2) okur |

---

## Veri Akış Haritası

### Sensör Verisi Akışı

1. **SensorSimulator** `generate_reading()` → `sensor_readings` tablosuna yazar (edge SQLite)
2. **EdgePipeline** `start_continuous_loop()` → `simulator.generate_reading()` + `simulator._write_db(sensor)` çağırır
3. **Grafana** `edge1_grafana:3001` → `EdgeDB` datasource → `/data/greenhouse.db` → `sensor_readings` tablosu → panel sorguları

### Alarm Üretim Akışı

1. `run_inference(sensor, image_b64)` çağrısı (gateway'den veya POST /predict'ten)
2. `asyncio.gather` ile 4 model **paralel** çağrılır: disease(:8101), irrigation(:8102), nutrition(:8103), anomaly(:8104)
3. Her model çıktısı `_call_monitor_all()` ile output_monitor(:8105) servisine gönderilir
4. `_build_fusion_input()` ile tek bir dict oluşturulur
5. `_call_fusion()` → fusion_engine(:8106) POST /evaluate → kural değerlendirilir → alarm dict
6. Fusion engine `alarms` tablosuna yazar (edge SQLite)
7. `_call_alarm()` → alarm_engine(:8107) POST /alarm → cooldown kontrolü → `alarms` tablosuna yazar
8. Alarm objesi gateway'e döner; POST /predict yanıtı olarak istemciye ulaşır

### Merkez Senkronizasyon Akışı

1. `central_aggregator` başlar, her 30 saniyede `sync_all_nodes()` çalışır
2. Her node için: `GET http://edge1_gateway:8100/alerts/unsynced` → unsynced alarmlar gelir
3. Gelen alarmlar `central.db`'nin `alarms` tablosuna `INSERT OR IGNORE` ile yazılır
4. `POST http://edge1_gateway:8100/alerts/sync-all` → edge'deki alarmlar `synced=1` yapılır
5. `node_registry` güncellenir: `last_ping`, `last_sync`, `status='ok'`
6. Merkez Grafana(:3000) → `CentralDB` datasource → `central.db` → `alarms` tablosu

### LLM Açıklama Akışı

1. `POST :9000/explain` ile bir Alarm objesi gönderilir
2. Central gateway → `POST central_llm:9003/explain` proxy eder
3. `LLMExplainer.explain(alarm)`:
   - `LLM_API_KEY` varsa: `POST https://api.openai.com/v1/chat/completions` (15s timeout)
   - Başarısız/yoksa: RULE_ID'ye göre Türkçe template seçilir
4. `_update_db_explanation()` → `central.db` `alarms` tablosunda `llm_explanation` alanı güncellenir
5. `{ alarm_id, explanation, source: "llm"|"template" }` yanıtı döner

---

## Kritik Konfigürasyon Dosyaları

### .env.example

| Değişken | Hangi Servis Okur | Açıklama | Örnek Değer |
|----------|-------------------|----------|-------------|
| `NODE_ID` | edge_gateway, output_monitor | Node tanımlayıcısı | `greenhouse_01` |
| `NODE_PROFILE` | (bilgi amaçlı) | Simulator profil adı | `node1` |
| `DISEASE_SERVICE_URL` | edge_gateway pipeline.py | Disease servis adresi | `http://edge1_disease:8101` |
| `IRRIGATION_SERVICE_URL` | edge_gateway pipeline.py | Irrigation servis adresi | `http://edge1_irrigation:8102` |
| `NUTRITION_SERVICE_URL` | edge_gateway pipeline.py | Nutrition servis adresi | `http://edge1_nutrition:8103` |
| `ANOMALY_SERVICE_URL` | edge_gateway pipeline.py | Anomaly servis adresi | `http://edge1_anomaly:8104` |
| `MONITOR_SERVICE_URL` | edge_gateway pipeline.py | Monitor servis adresi | `http://edge1_output_monitor:8105` |
| `FUSION_SERVICE_URL` | edge_gateway pipeline.py + main.py | Fusion servis adresi | `http://edge1_fusion:8106` |
| `ALARM_SERVICE_URL` | edge_gateway pipeline.py + main.py | Alarm servis adresi | `http://edge1_alarm:8107` |
| `CENTRAL_API_URL` | (şu an kullanılmıyor, edge'de) | Merkez gateway | `http://central_gateway:9000` |
| `DB_PATH` | sensor_simulator, output_monitor, alarm_engine, fusion_engine | SQLite dosya yolu | `/data/greenhouse.db` |
| `SIMULATOR_CONFIG_PATH` | edge_gateway pipeline.py | Node YAML tam yol | `/app/simulator/config/node1.yaml` |
| `DATASET_PATH` | edge_gateway pipeline.py | PlantVillage dataset | `` (boş = image sim kapalı) |
| `SYNC_INTERVAL_SECONDS` | (alarm_engine'de kullanılmıyor; CENTRAL_SYNC_INTERVAL kullanılır) | Edge sync aralığı | `30` |
| `ALARM_COOLDOWN_SECONDS` | alarm_engine | Aynı kural tekrar süresi | `60` |
| `NODE1_GATEWAY_URL` | central_aggregator | Edge 1 gateway URL | `http://edge1_gateway:8100` |
| `NODE2_GATEWAY_URL` | central_aggregator | Edge 2 gateway URL | `http://edge2_gateway:8200` |
| `CENTRAL_SYNC_INTERVAL` | central_aggregator | Merkez sync aralığı | `30` |
| `CENTRAL_DB_PATH` | central_aggregator, central_lstm, central_llm | Merkez SQLite yolu | `/data/central.db` |
| `LLM_API_KEY` | central_llm explainer.py | OpenAI API anahtarı | `sk-...` |
| `LLM_MODEL` | central_llm explainer.py | Kullanılacak model | `gpt-4o-mini` |
| `LLM_MAX_TOKENS` | central_llm explainer.py | Maksimum token | `200` |
| `LOG_LEVEL` | Tüm servisler | Loglama seviyesi | `INFO` |

### simulator/config/node1.yaml

| Alan | Açıklama |
|------|----------|
| `profile` | Dahili profil adı (`node1`) |
| `node_id` | Alarm ve DB'deki node tanımlayıcısı (`greenhouse_01`) |
| `name` | İnsan okunabilir ad (`Sera A - Domates`) |
| `sensor_profile.<sensör>.mean` | Gaussian dağılım merkezi |
| `sensor_profile.<sensör>.std` | Gaussian gürültü standart sapması |
| `sensor_profile.<sensör>.min/max` | Değer kırpma sınırları |
| `image_class_distribution` | Sınıf olasılıkları (toplam = 1.0 olmalı) |
| `emit_interval_seconds` | Sürekli döngü bekleme süresi |
| `anomaly_injection.enabled` | `false` = otomatik anomali yok |
| `anomaly_injection.probability` | `0.0` = spike olasılığı |

### edge/fusion_engine/rules/rules.yaml

**Yapı:**
```yaml
rules:
  - id: "RULE_XXX"           # Benzersiz kimlik (alarm rule_id alanına gider)
    name: "snake_case_name"  # İnsan okunabilir
    conditions:              # Tüm koşullar AND'dir
      - field: "dot.separated.path"   # FusionInput'ta nested path
        operator: "gt|lt|eq|neq|gte|lte"
        value: <number veya boolean veya string>
    action:
      alarm_level: "INFO|WARNING|HIGH|CRITICAL"
      message_key: "template_anahtar"   # LLM explainer kullanır
```

**Örnek kural analizi (RULE_001):**
- Koşul 1: `sensor_reading.sensors.humidity > 80` → Nem %80'den fazla
- Koşul 2: `disease_output.top_confidence > 0.85` → Disease modeli %85 güvenle hastalık tespiti
- Koşul 3: `anomaly_output.is_anomaly = false` → Sensör anomalisi yok (gerçek hastalık)
- Aksiyon: CRITICAL alarm, `fungal_risk_critical` mesaj anahtarı

**Önemli**: Kurallar ilk eşleşmede durur (short-circuit). RULE_001'den RULE_010'a sıra önemli.

### monitoring/grafana/provisioning/datasources/

| Dosya | uid | DB Yolu | Kullanıcı Dashboard |
|-------|-----|---------|---------------------|
| `edge_datasource.yaml` | `EdgeDB` | `/data/greenhouse.db` | `edge_node.json` tüm panelleri |
| `central_datasource.yaml` | `CentralDB` | `/data/central.db` | `central.json` tüm panelleri |

**Not**: `edge_datasource.yaml`'da `isDefault: false`. Merkez Grafana bu edge datasource'u da alır (aynı imaj), ancak `/data/greenhouse.db` üzerine `central_db` volume bağlıdır → dosya adı farklı (`central.db`). Bu nedenle **EdgeDB datasource merkez Grafana'da veri göstermez**.

---

## Servisler Arası Bağımlılık Haritası

**edge1_gateway (8100)**
- Çağırdığı servisler: disease:8101, irrigation:8102, nutrition:8103, anomaly:8104, output_monitor:8105, fusion:8106, alarm:8107
- Çağrıldığı servisler: kullanıcı/test, central_aggregator (unsynced pull için)
- Okuduğu DB tabloları: `sensor_readings`
- Yazdığı DB tabloları: `sensor_readings` (simulator üzerinden)

**edge1_disease (8101)**
- Çağırdığı servisler: Yok
- Çağrıldığı servisler: edge1_gateway
- DB: Yok

**edge1_irrigation (8102) / edge1_nutrition (8103) / edge1_anomaly (8104)**
- Aynı patern: gateway tarafından çağrılır, DB yok

**edge1_output_monitor (8105)**
- Çağırdığı servisler: Yok
- Çağrıldığı servisler: edge1_gateway
- Yazdığı DB tabloları: `monitor_events`

**edge1_fusion (8106)**
- Çağırdığı servisler: Yok
- Çağrıldığı servisler: edge1_gateway
- Yazdığı DB tabloları: `alarms`

**edge1_alarm (8107)**
- Çağırdığı servisler: Yok
- Çağrıldığı servisler: edge1_gateway
- Okuduğu DB tabloları: `alarms`
- Yazdığı DB tabloları: `alarms`

**central_aggregator (9001)**
- Çağırdığı servisler: edge1_gateway (GET /alerts/unsynced, POST /alerts/sync-all), edge2_gateway
- Çağrıldığı servisler: central_gateway, startup sırasında kendi kendine
- Okuduğu DB tabloları: `alarms`, `node_registry`
- Yazdığı DB tabloları: `alarms`, `node_registry`

**central_lstm (9002)**
- Çağırdığı servisler: Yok
- Çağrıldığı servisler: central_gateway
- Okuduğu DB tabloları: `sensor_readings` (central DB)

**central_llm (9003)**
- Çağırdığı servisler: OpenAI API (harici)
- Çağrıldığı servisler: central_gateway
- Yazdığı DB tabloları: `alarms` (llm_explanation alanı)

**central_gateway (9000)**
- Çağırdığı servisler: central_aggregator:9001, central_lstm:9002, central_llm:9003
- Çağrıldığı servisler: kullanıcı/test
- DB: Yok (sadece proxy)

---

## Bilinen Sorunlar ve Açık Noktalar

### Hardcode Kalan Değerler
1. **`central/lstm_yield/api/main.py:36`** → `KNOWN_NODES = ["greenhouse_01", "greenhouse_02"]` hardcode. Yeni node eklenirse buranın da güncellenmesi gerekir.
2. **`central/lstm_yield/train.py:103`** → `node_ids = ["greenhouse_01", "greenhouse_02"]` hardcode. `KNOWN_NODES` ile aynı sorun.

### Eksik Implementasyonlar (Stub'lar)
3. **`central/aggregator/aggregator.py`** → `aggregate_nodes()` her zaman `None` döner; bu dosya hiçbir servis tarafından import edilmiyor (gerçek mantık `service.py`'de). Kafa karışıklığına yol açabilir.
4. **`central/lstm_yield/lstm.py`** → `forecast()` her zaman `None` döner; bu dosya da import edilmiyor.

### DB Tutarsızlıkları
5. **`model_outputs` tablosu** → `database/schema.sql`'de tanımlanmış ama **hiçbir servis bu tabloya yazmıyor**. Şema ile uygulama arasında açıklık var; Grafana paneli bu tablodan veri okumaya çalışırsa boş gelir.
6. **`alarms` tablosu** hem `edge/fusion_engine/api/main.py`'de hem `edge/alarm_engine/engine.py`'de `CREATE TABLE IF NOT EXISTS` ile oluşturuluyor. Bu idempotent olduğu için sorun çıkarmaz ama gereksiz tekrar var.

### Grafana Veri Sorunu (Neden Veri Gelmez?)
7. **EdgeDB datasource - Merkez Grafana'da çalışmaz**: `edge_datasource.yaml` tüm Grafana build'lerinde kopyalanıyor. Merkez Grafana'ya bağlı `central_db` volume `/data/central.db` dosyasını barındırıyor. `EdgeDB` datasource `/data/greenhouse.db`'yi arıyor — bu dosya merkez container'da yok. Edge dashboard (`edge_node.json`) `EdgeDB` uid'ini kullanıyor; merkez Grafana'da bu dashboard açılırsa veri gelmez.
8. **`edge_datasource.yaml` `isDefault: false`** — edge Grafana'larda da `isDefault: false`. `edge_node.json` dashboard'u panel başına `datasource uid: EdgeDB` belirtse sorun olmaz, ama uid belirtilmemişse Grafana varsayılan datasource'u kullanır ki bu da `CentralDB`.
9. **Monitor buffer sıfırlanır**: `output_monitor` servisi yeniden başlatıldığında tüm buffer bellekten silinir; z-score hesaplaması için `MIN_SAMPLES=10` dolana kadar `is_anomaly=false` döner.

### Test Coverage Eksiklikleri
10. **`tests/integration/test_full_system.py`** — Dosya var ama içeriği görülmedi; Phase 5-6 entegrasyon testleri var mı doğrulanmadı.
11. **Disease modeli testleri** — Gerçek PlantVillage görüntüleri olmadan F1 hedefi (≥0.85) sentetik verilerle karşılanmayabilir (train.py'de de uyarı var).

### Diğer Dikkat Noktaları
12. **`rules.yaml` kalıcılığı**: `update_threshold()` `rules.yaml` dosyasına yazar, ancak container yeniden başlatılırsa Docker build sırasındaki orijinal hali geri yüklenir (volume mount yok). Threshold değişiklikleri yeniden başlatmada kaybolur.
13. **Image simulator opsiyonel**: `DATASET_PATH` boşsa disease modeli her çağrıda fallback değer (`unknown`, confidence=0.0) döner. Bu durumda RULE_001 (fungal risk) asla tetiklenemez çünkü `top_confidence=0.0 < 0.85`.
14. **`SYNC_INTERVAL_SECONDS`** env değişkeni `.env.example`'da ve `x-edge1-env` YAML anchor'ında tanımlanmış ama gerçekte hiçbir servis bu değişkeni kullanmıyor; `CENTRAL_SYNC_INTERVAL` kullanılıyor.
