import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="RainPredictor Pro",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Kaggle Avustralya Veri Seti Popülasyon İstatistikleri (Z-Skor Dönüşümü İçin)
# Modelin sayıları doğru anlayabilmesi için ham veriyi bu istatistiklerle scale edeceğiz
METADATA = {
    'Humidity3pm': {'mean': 51.48, 'std': 20.75},
    'Pressure3pm': {'mean': 1015.25, 'std': 7.03},
    'Sunshine': {'mean': 7.61, 'std': 3.78},
    'Cloud3pm': {'mean': 4.50, 'std': 2.43},
    'MaxTemp': {'mean': 23.22, 'std': 7.12},
    'MinTemp': {'mean': 12.19, 'std': 6.40},
    'WindGustSpeed': {'mean': 40.03, 'std': 13.60},
    'Rainfall': {'mean': 2.36, 'std': 8.47},
    'Temp_Range': {'mean': 11.03, 'std': 4.58},
    'Humidity_Temp_Interaction': {'mean': 1195.36, 'std': 485.20}
}


def apply_standard_scaler(feature_name, raw_value):
    """Ham değeri modelin anlayacağı Z-Skoruna (StandardScaler) dönüştürür"""
    mean = METADATA[feature_name]['mean']
    std = METADATA[feature_name]['std']
    return (raw_value - mean) / std


# 3. Load Model Safely
@st.cache_resource
def load_resources():
    try:
        model = joblib.load("models/xgboost_rain_model.pkl")
    except:
        model = joblib.load("../models/xgboost_rain_model.pkl")
    features = model.get_booster().feature_names
    return model, features


try:
    loaded_xgb, feature_names = load_resources()
except Exception as e:
    st.error("⚠️ Model file could not be found.")
    st.stop()

# 4. App Header
st.title("⛈️ RainPredictor Pro: Interactive Weather Simulation")
st.caption(
    "Adjust the parameters below. Girdiler otomatik olarak StandardScaler işleminden geçirilerek XGBoost modeline iletilir.")
st.write("---")

# 5. Layout
col_sidebar, col_dashboard = st.columns([1, 2], gap="large")

with col_sidebar:
    st.subheader("🎛️ Atmospheric Controls")

    st.markdown("**📍 Geographical Context**")
    location_choice = st.selectbox("Select Location",
                                   ["Sydney", "Melbourne", "Brisbane", "Adelaide", "Perth", "Cairns"])
    wind_choice = st.selectbox("Wind Gust Direction", ["W", "E", "N", "S", "NW", "NE", "SW", "SE"])

    st.markdown("**🌤️ Core Thermodynamic Indicators**")
    # Varsayılan olarak tertemiz, güneşli bir günle başlatıyoruz
    humidity_3pm = st.slider("3pm Humidity (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    pressure_3pm = st.slider("3pm Atmospheric Pressure (hPa)", min_value=980.0, max_value=1040.0, value=1028.0,
                             step=0.5)
    sunshine = st.slider("Sunshine Duration (Hours)", min_value=0.0, max_value=15.0, value=11.0, step=0.5)
    cloud_3pm = st.slider("3pm Cloud Cover (Oktas)", min_value=0.0, max_value=8.0, value=1.0, step=1.0)

    st.markdown("**%🌡️ Temperature & Wind Speed**")
    max_temp = st.slider("Max Temperature (°C)", min_value=-5.0, max_value=50.0, value=25.0, step=0.5)
    min_temp = st.slider("Min Temperature (°C)", min_value=-5.0, max_value=40.0, value=14.0, step=0.5)
    wind_gust = st.slider("Wind Gust Speed (km/h)", min_value=0.0, max_value=130.0, value=25.0, step=1.0)
    rainfall_today = st.slider("Today's Recorded Rainfall (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)

    # 6. Build Scenario Dataframe
    scenario_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    # Sayısal Değerleri Matematiksel Olarak Ölçekleyip (Z-Score) DataFrame'e Yaz
    scenario_df['Humidity3pm'] = apply_standard_scaler('Humidity3pm', humidity_3pm)
    scenario_df['Pressure3pm'] = apply_standard_scaler('Pressure3pm', pressure_3pm)
    scenario_df['Sunshine'] = apply_standard_scaler('Sunshine', sunshine)
    scenario_df['Cloud3pm'] = apply_standard_scaler('Cloud3pm', cloud_3pm)
    scenario_df['MaxTemp'] = apply_standard_scaler('MaxTemp', max_temp)
    scenario_df['MinTemp'] = apply_standard_scaler('MinTemp', min_temp)
    scenario_df['WindGustSpeed'] = apply_standard_scaler('WindGustSpeed', wind_gust)
    scenario_df['Rainfall'] = apply_standard_scaler('Rainfall', rainfall_today)

    # Etkileşimli Özellikleri Hesapla ve Onları da Ölçekle
    temp_range_raw = max_temp - min_temp
    interaction_raw = humidity_3pm * max_temp
    scenario_df['Temp_Range'] = apply_standard_scaler('Temp_Range', temp_range_raw)
    scenario_df['Humidity_Temp_Interaction'] = apply_standard_scaler('Humidity_Temp_Interaction', interaction_raw)

    # One-Hot Encoded (Kategorik) Alanları Doldur (Kategorik alanlar scale edilmez, 0 veya 1 kalır)
    loc_col = f"Location_{location_choice}"
    wind_col = f"WindGustDir_{wind_choice}"
    if loc_col in feature_names:
        scenario_df[loc_col] = 1.0
    if wind_col in feature_names:
        scenario_df[wind_col] = 1.0

    # Sıralamayı modelin beklediği formata sok
    scenario_df = scenario_df[feature_names]

with col_dashboard:
    st.subheader("📊 Real-Time AI Visualizations")

    # Ölçeklenmiş veriyle nihai tahmini gerçekleştir
    prob_rain = loaded_xgb.predict_proba(scenario_df)[0][1]
    binary_decision = loaded_xgb.predict(scenario_df)[0]

    # Skor Kartları
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label="AI Predicted Probability of Rain", value=f"{prob_rain * 100:.1f}%")
    with col_m2:
        decision_text = "🌧️ YES (Rain Expected)" if binary_decision == 1 else "☀️ NO (Dry / Clear)"
        st.metric(label="Final Binary Decision", value=decision_text)

    st.write("---")

    # Hız Göstergesi Grafiği
    st.markdown("### 🔮 Predictive Confidence Speedometer")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_rain * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Precipitation Probability Scale for {location_choice}", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "#38bdf8" if prob_rain < 0.5 else "#f43f5e"},
            'steps': [
                {'range': [0, 35], 'color': '#a7f3d0'},
                {'range': [35, 65], 'color': '#fde68a'},
                {'range': [65, 100], 'color': '#fecdd3'}
            ],
        }
    ))
    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Özet Bilgi Paneli
    st.markdown("### 💡 Scaling Pipeline Log (Technical Verification)")
    st.info(f"💡 **Data Pipeline Note:** Sliders'dan gelen ham değerler arka planda Z-Score formatına dönüştürüldü. "
            f"Örneğin, seçtiğiniz %{humidity_3pm} nem değeri modele **{scenario_df['Humidity3pm'].values[0]:.2f}** Z-Skoru olarak iletildi. "
            f"Bu sayede modeliniz tam eğitim anındaki doğrulukla çalışmaktadır.")