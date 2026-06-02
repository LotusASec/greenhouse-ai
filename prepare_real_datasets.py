import pandas as pd
import numpy as np
from pathlib import Path

SOURCE_FILE = "plant_health_data (1).csv"
TARGET_DIR = Path("models/shared/data")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

print("⏳ Tüm veri setleri çift sütun güvencesiyle senkronize ediliyor...")

# 1. Gerçek ham veriyi yükle
df_raw = pd.read_csv(SOURCE_FILE)

# 2. Girdi özelliklerini projenin frozen şablonuna dönüştür
df_mapped = pd.DataFrame()
df_mapped['temperature'] = df_raw['Ambient_Temperature']
df_mapped['humidity'] = df_raw['Humidity']
df_mapped['soil_moisture'] = df_raw['Soil_Moisture']
df_mapped['light'] = df_raw['Light_Intensity']
df_mapped['ec'] = df_raw['Electrochemical_Signal'] * 2.0
df_mapped['ph'] = df_raw['Soil_pH']

# --- A. ANOMALİ MODELİ VERİSİ ---
rng_train = np.random.RandomState(44)
rng_test  = np.random.RandomState(45)
rng_inj   = np.random.RandomState(46)
n_normal, n_test_normal, n_each = 2000, 200, 5

def _gen_normal(n, rng):
    return pd.DataFrame({
        "temperature":   rng.uniform(18.0, 35.0, n),
        "humidity":      rng.uniform(40.0, 90.0, n),
        "soil_moisture": rng.uniform(20.0, 80.0, n),
        "light":         rng.uniform(0.0, 1000.0, n),
        "ec":            rng.uniform(1.0,  3.5, n),
        "ph":            rng.uniform(5.5,  7.0, n),
    })

_gen_normal(n_normal, rng_train).to_csv(TARGET_DIR / "anomaly_normal.csv", index=False)
_gen_normal(n_test_normal, rng_test).to_csv(TARGET_DIR / "anomaly_test_normal.csv", index=False)

frames = []
f1 = pd.DataFrame({"temperature": rng_inj.uniform(52.0, 65.0, n_each), "humidity": rng_inj.uniform(2.0, 8.0, n_each), "soil_moisture": rng_inj.uniform(1.0, 5.0, n_each), "light": rng_inj.uniform(900.0, 1200.0, n_each), "ec": rng_inj.uniform(2.0, 2.5, n_each), "ph": rng_inj.uniform(6.0, 6.5, n_each)})
frames.append(f1)
f2 = pd.DataFrame({"temperature": rng_inj.uniform(50.0, 65.0, n_each), "humidity": rng_inj.uniform(2.0, 8.0, n_each), "soil_moisture": rng_inj.uniform(54.0, 56.0, n_each), "light": rng_inj.uniform(695.0, 705.0, n_each), "ec": rng_inj.uniform(8.0, 12.0, n_each), "ph": rng_inj.uniform(1.5, 3.0, n_each)})
frames.append(f2)
f3 = pd.DataFrame({"temperature": rng_inj.uniform(55.0, 70.0, n_each), "humidity": rng_inj.uniform(2.0, 8.0, n_each), "soil_moisture": rng_inj.uniform(1.0, 5.0, n_each), "light": rng_inj.uniform(695.0, 705.0, n_each), "ec": rng_inj.uniform(7.0, 10.0, n_each), "ph": rng_inj.uniform(6.2, 6.4, n_each)})
frames.append(f3)
f4 = pd.DataFrame({"temperature": rng_inj.uniform(48.0, 48.0, n_each), "humidity": rng_inj.uniform(95.0, 105.0, n_each), "soil_moisture": rng_inj.uniform(90.0, 100.0, n_each), "light": rng_inj.uniform(1800.0, 2200.0, n_each), "ec": rng_inj.uniform(7.0, 10.0, n_each), "ph": rng_inj.uniform(10.0, 13.0, n_each)})
frames.append(f4)
f5 = pd.DataFrame({"temperature": rng_inj.uniform(1.0, 5.0, n_each), "humidity": rng_inj.uniform(1.0, 5.0, n_each), "soil_moisture": rng_inj.uniform(1.0, 5.0, n_each), "light": rng_inj.uniform(0.0, 1.0, n_each), "ec": rng_inj.uniform(0.0, 0.1, n_each), "ph": rng_inj.uniform(1.0, 2.0, n_each)})
frames.append(f5)
pd.concat(frames, ignore_index=True).to_csv(TARGET_DIR / "anomaly_test_inject.csv", index=False)

# --- B. SULAMA MODELİ ---
df_irr = df_mapped.copy()
df_irr['irrigate'] = ((df_raw['Soil_Moisture'] < 25.0) | ((df_raw['Plant_Health_Status'] == 'High Stress') & (df_raw['Soil_Moisture'] < 35.0))).astype(int)
df_irr['amount_liters'] = np.where(df_irr['irrigate'] == 1, np.round((45 - df_irr['soil_moisture']) * 0.7 + (df_irr['temperature'] * 0.15), 2), 0.0)
df_irr.sample(frac=0.8, random_state=44).to_csv(TARGET_DIR / "irrigation_train.csv", index=False)
df_irr.sample(frac=0.2, random_state=44).to_csv(TARGET_DIR / "irrigation_test.csv", index=False)

# --- C. GÜBRELEME/BESİN MODELİ (HATA ÖNLEYİCİ: ÇİFT ETİKET) ---
df_nut = df_mapped.copy()
conditions = [
    (df_raw['Nitrogen_Level'] < 15.0), 
    (df_raw['Phosphorus_Level'] < 30.0), 
    (df_raw['Potassium_Level'] < 25.0)
]
calculated_labels = np.select(conditions, [0, 1, 2], default=3)

# Kod hangisini isterse istesin hata vermemesi için iki ismi de sütun olarak ekliyoruz!
df_nut['deficiency_class'] = calculated_labels
df_nut['label'] = calculated_labels

df_nut.sample(frac=0.8, random_state=44).to_csv(TARGET_DIR / "nutrition_train.csv", index=False)
df_nut.sample(frac=0.2, random_state=44).to_csv(TARGET_DIR / "nutrition_test.csv", index=False)

print("🎉 MÜKEMMEL! Tüm şemalar ve çift etiket güvencesi tamamlandı.")