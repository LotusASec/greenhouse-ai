import pandas as pd
import numpy as np
from pathlib import Path

# Dosya yolları
SOURCE_FILE = "plant_health_data (1).csv"
TARGET_DIR = Path("models/shared/data")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

print("⏳ Gerçek veri seti projenin TR-11 standartlarına göre işleniyor...")

# 1. Ham veriyi yükle
df_raw = pd.read_csv(SOURCE_FILE)

# 2. Sütun isimlerini projenin dondurulmuş (frozen) şablonuna dönüştür
df_mapped = pd.DataFrame()
df_mapped['temperature'] = df_raw['Ambient_Temperature']
df_mapped['humidity'] = df_raw['Humidity']
df_mapped['soil_moisture'] = df_raw['Soil_Moisture']
df_mapped['light'] = df_raw['Light_Intensity']
df_mapped['ec'] = df_raw['Electrochemical_Signal'] * 2.0  # Sinyali standart EC aralığına ölçekle
df_mapped['ph'] = df_raw['Soil_pH']

# --- A. ANOMALİ MODELİ VERİ SETLERİ ---
# Sağlıklı ve Hafif Stresli durumları normal kabul edip eğitiyoruz
df_normal = df_mapped[df_raw['Plant_Health_Status'] != 'High Stress'].copy()
df_high_stress = df_mapped[df_raw['Plant_Health_Status'] == 'High Stress'].copy()

# Verileri karıştır ve böl
df_normal = df_normal.sample(frac=1, random_state=42).reset_index(drop=True)
train_size = int(len(df_normal) * 0.8)

df_normal.iloc[:train_size].to_csv(TARGET_DIR / "anomaly_normal.csv", index=False)
df_normal.iloc[train_size:].to_csv(TARGET_DIR / "anomaly_test_normal.csv", index=False)

# Anomali testi için High Stress verilerini alalım ve içine birkaç sert sensör sapması (Spike/Freeze) enjekte edelim
df_inject = df_high_stress.sample(n=100, random_state=42).copy()
df_inject.iloc[0:20, df_inject.columns.get_loc('temperature')] = 95.0  # Sıcaklık patlaması
df_inject.iloc[30:50, df_inject.columns.get_loc('light')] = 0.0       # Gece vakti değilken ışık sönmesi
df_inject.to_csv(TARGET_DIR / "anomaly_test_inject.csv", index=False)


# --- B. SULAMA (IRRIGATION) MODELİ VERİ SETLERİ ---
# Kural: Toprak nemi düştükçe ve bitki strese girdiyse sulama gerekir
df_irr = df_mapped.copy()
df_irr['irrigate'] = ((df_raw['Soil_Moisture'] < 25.0) | 
                      ((df_raw['Plant_Health_Status'] == 'High Stress') & (df_raw['Soil_Moisture'] < 35.0))).astype(int)

# Sulama miktarı (Regressor için litre hesabı)
df_irr['recommended_amount_liters'] = np.where(
    df_irr['irrigate'] == 1,
    np.round((45 - df_irr['soil_moisture']) * 0.7 + (df_irr['temperature'] * 0.15), 2),
    0.0
)

# %80 Train, %20 Test olarak böl
df_irr_train = df_irr.sample(frac=0.8, random_state=42)
df_irr_test = df_irr.drop(df_irr_train.index)
df_irr_train.to_csv(TARGET_DIR / "irrigation_train.csv", index=False)
df_irr_test.to_csv(TARGET_DIR / "irrigation_test.csv", index=False)


# --- C. GÜBRELEME (NUTRITION) MODELİ VERİ SETLERİ ---
# 4 Sınıf Kurgusu: 0: Azot Eksik, 1: Fosfor Eksik, 2: Potasyum Eksik, 3: Dengeli (Healthy)
df_nut = df_mapped.copy()

conditions = [
    (df_raw['Nitrogen_Level'] < 15.0),
    (df_raw['Phosphorus_Level'] < 30.0),
    (df_raw['Potassium_Level'] < 25.0)
]
choices = [0, 1, 2]
df_nut['label'] = np.select(conditions, choices, default=3)

# %80 Train, %20 Test olarak böl
df_nut_train = df_nut.sample(frac=0.8, random_state=42)
df_nut_test = df_nut.drop(df_nut_train.index)
df_nut_train.to_csv(TARGET_DIR / "nutrition_train.csv", index=False)
df_nut_test.to_csv(TARGET_DIR / "nutrition_test.csv", index=False)

print("🎉 Başarılı! Tüm gerçek veri setleri 'models/shared/data/' klasörüne projenin tam istediği formatta dağıtıldı!")