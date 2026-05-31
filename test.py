import pandas as pd
import numpy as np
import joblib

print("=== SAFE INFERENCE CODE-ONLY TEST ===")

# 1. Modeli yükle ve beklediği ham özellik isimlerini al
try:
    model = joblib.load("models/xgboost_rain_model.pkl")
    feature_names = model.get_booster().feature_names
    print(f"✅ Model successfully loaded. Expected features count: {len(feature_names)}")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    exit()

# 2. Tamamen sıfırlardan oluşan bir şablon satır oluştur (One-hot alanları 0 kalacak)
mock_data = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

# 3. SENARYO A: Kesin Güneşli Gün Girdileri
mock_data['Humidity3pm'] = 15.0       # Çok düşük nem
mock_data['Pressure3pm'] = 1030.0     # Çok yüksek basınç
mock_data['Sunshine'] = 12.0          # Bol güneş
mock_data['Cloud3pm'] = 0.0           # Sıfır bulut
mock_data['MaxTemp'] = 28.0
mock_data['MinTemp'] = 16.0
mock_data['WindGustSpeed'] = 20.0
mock_data['Rainfall'] = 0.0
mock_data['Temp_Range'] = 28.0 - 16.0
mock_data['Humidity_Temp_Interaction'] = 15.0 * 28.0

# Özellik sıralamasını modelin tam istediği hale getir
mock_data = mock_data[feature_names]

# Tahmin Et (Güneşli)
prob_sunny = model.predict_proba(mock_data)[0][1]
decision_sunny = model.predict(mock_data)[0]

print("\n☀️ [PURE CODE] SUNNY SCENARIO RESULT:")
print(f"-> Predicted Rain Probability: {prob_sunny * 100:.2f}%")
print(f"-> Binary Decision: {decision_sunny}")


# 4. SENARYO B: Kesin Fırtınalı Gün Girdileri
mock_data_storm = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
mock_data_storm['Humidity3pm'] = 95.0     # Tavan yapmış nem
mock_data_storm['Pressure3pm'] = 990.0     # Çökmüş basınç
mock_data_storm['Sunshine'] = 0.0          # Sıfır güneş
mock_data_storm['Cloud3pm'] = 8.0          # Tamamen kapalı gökyüzü
mock_data_storm['MaxTemp'] = 18.0
mock_data_storm['MinTemp'] = 14.0
mock_data_storm['WindGustSpeed'] = 75.0
mock_data_storm['Rainfall'] = 12.5
mock_data_storm['Temp_Range'] = 18.0 - 14.0
mock_data_storm['Humidity_Temp_Interaction'] = 95.0 * 18.0

# Sıralamayı eşitle
mock_data_storm = mock_data_storm[feature_names]

# Tahmin Et (Fırtına)
prob_storm = model.predict_proba(mock_data_storm)[0][1]
decision_storm = model.predict(mock_data_storm)[0]

print("\n⛈️ [PURE CODE] STORMY SCENARIO RESULT:")
print(f"-> Predicted Rain Probability: {prob_storm * 100:.2f}%")
print(f"-> Binary Decision: {decision_storm}")
print("=====================================")
