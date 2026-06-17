import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="RainPredictor Pro",
    page_icon="🌥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Load Model and Scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/xgboost_rain_model.pkl")
    except:
        model = joblib.load("../models/xgboost_rain_model.pkl")
    features = model.get_booster().feature_names
    return model, features

@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load("models/scaler.pkl")
    except:
        scaler = joblib.load("../models/scaler.pkl")
    return scaler

try:
    loaded_xgb, feature_names = load_model()
    scaler = load_scaler()
    st.success("✅ Model and scaler successfully loaded.")
except Exception as e:
    st.error(f"⚠️ Could not load model or scaler: {e}")
    st.stop()

# 3. Title
st.title("🌥️ RainPredictor Pro: Interactive Weather Simulation")
st.caption("A borderline/uncertain day is set by default. You can change the values to see instant predictions.")
st.write("---")

# 4. Interface Layout
col_sidebar, col_dashboard = st.columns([1, 2], gap="large")

with col_sidebar:
    st.subheader("🎛️ Atmospheric Controls")

    st.markdown("**📍 Geographic & Temporal Context**")
    location_choice = st.selectbox("Select Location",
                                   ["Sydney", "Melbourne", "Brisbane", "Adelaide", "Perth", "Cairns",
                                    "Canberra", "Darwin", "Hobart", "AliceSprings"])
    month_choice = st.selectbox("Month of the Year",
                                list(range(1, 13)),
                                format_func=lambda m: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][m - 1])

    month_sin = np.sin(2 * np.pi * month_choice / 12.0)
    month_cos = np.cos(2 * np.pi * month_choice / 12.0)

    st.markdown("**🌬️ Wind Directions**")
    wind_dir_options = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    wind_gust_dir = st.selectbox("Wind Gust Direction (Strongest gust)", wind_dir_options, index=9)
    wind_dir_9am = st.selectbox("Wind Direction at 9 AM", wind_dir_options, index=11)
    wind_dir_3pm = st.selectbox("Wind Direction at 3 PM", wind_dir_options, index=9)

    st.markdown("**🌤️ Core Meteorological Indicators**")
    humidity_3pm = st.slider("3pm Humidity (%)", 0.0, 100.0, 65.0, 1.0)
    pressure_9am = st.slider("9am Pressure (hPa)", 980.0, 1040.0, 1025.0, 0.5)
    pressure_3pm = st.slider("3pm Pressure (hPa)", 980.0, 1040.0, 1012.0, 0.5)
    sunshine = st.slider("Sunshine Duration (Hours)", 0.0, 15.0, 4.0, 0.5)
    cloud_9am = st.slider("9am Cloud Cover (Oktas)", 0.0, 8.0, 0.0, 1.0)
    cloud_3pm = st.slider("3pm Cloud Cover (Oktas)", 0.0, 8.0, 4.0, 1.0)

    st.markdown("**🌡️ Temperature & Wind Speed**")
    max_temp = st.slider("Max Temperature (°C)", -5.0, 50.0, 22.0, 0.5)
    min_temp = st.slider("Min Temperature (°C)", -5.0, 40.0, 14.0, 0.5)
    temp_9am = st.slider("9am Temperature (°C)", -5.0, 45.0, 18.0, 0.5)
    temp_3pm = st.slider("3pm Temperature (°C)", -5.0, 45.0, 24.0, 0.5)
    wind_speed_9am = st.slider("Wind Speed at 9am (km/h)", 0.0, 100.0, 10.0, 1.0)
    wind_speed_3pm = st.slider("Wind Speed at 3pm (km/h)", 0.0, 100.0, 12.0, 1.0)
    wind_gust = st.slider("Max Wind Gust Speed (km/h)", 0.0, 130.0, 25.0, 1.0)

    st.markdown("**💧 Rainfall & Humidity**")
    rainfall_today = st.slider("Today's Rainfall (mm)", 0.0, 100.0, 0.0, 0.1)
    humidity_9am = st.slider("9am Humidity (%)", 0.0, 100.0, 30.0, 1.0)
    evaporation = st.slider("Evaporation (mm)", 0.0, 20.0, 8.0, 0.1)

    rain_today = st.radio("Did it rain today?", ["No", "Yes"])
    rain_today_binary = 1.0 if rain_today == "Yes" else 0.0

    # 5. Create Scenario DataFrame with Raw Values
    scenario_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    scenario_df['MinTemp'] = min_temp
    scenario_df['MaxTemp'] = max_temp
    scenario_df['Rainfall'] = rainfall_today
    scenario_df['Evaporation'] = evaporation
    scenario_df['Sunshine'] = sunshine
    scenario_df['WindGustSpeed'] = wind_gust
    scenario_df['WindSpeed9am'] = wind_speed_9am
    scenario_df['WindSpeed3pm'] = wind_speed_3pm
    scenario_df['Humidity9am'] = humidity_9am
    scenario_df['Humidity3pm'] = humidity_3pm
    scenario_df['Pressure9am'] = pressure_9am
    scenario_df['Pressure3pm'] = pressure_3pm
    scenario_df['Cloud9am'] = cloud_9am
    scenario_df['Cloud3pm'] = cloud_3pm
    scenario_df['Temp9am'] = temp_9am
    scenario_df['Temp3pm'] = temp_3pm

    # Derived Features
    scenario_df['Temp_Range'] = max_temp - min_temp
    scenario_df['Humidity_Temp_Interaction'] = humidity_3pm * max_temp
    scenario_df['Pressure_Drop'] = pressure_9am - pressure_3pm

    # Seasonality
    scenario_df['Month_Sin'] = month_sin
    scenario_df['Month_Cos'] = month_cos

    # Rain Today
    scenario_df['RainToday'] = rain_today_binary

    # One-Hot Categorical Columns
    for col in [f"Location_{location_choice}",
                f"WindGustDir_{wind_gust_dir}",
                f"WindDir9am_{wind_dir_9am}",
                f"WindDir3pm_{wind_dir_3pm}"]:
        if col in feature_names:
            scenario_df[col] = 1.0

    # Reorder columns as expected by the model
    scenario_df = scenario_df[feature_names]

    # Scaler should only scale columns except RainToday
    scaler_features = [f for f in feature_names if f != 'RainToday']
    scenario_scaled = scenario_df.copy()
    scenario_scaled[scaler_features] = scaler.transform(scenario_df[scaler_features])

with col_dashboard:
    st.subheader("📊 Real-Time AI Visualizations")

    prob_rain = loaded_xgb.predict_proba(scenario_scaled)[0][1]
    binary_decision = loaded_xgb.predict(scenario_scaled)[0]

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label="AI Rain Probability Prediction", value=f"{prob_rain * 100:.1f}%")
    with col_m2:
        decision_text = "🌧️ YES (Rain Expected)" if binary_decision == 1 else "☀️ NO (Dry / Clear)"
        st.metric(label="Final Decision", value=decision_text)

    st.write("---")

    # Gauge Chart
    st.markdown("### 🔮 Prediction Confidence Indicator")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_rain * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Rain Probability Scale - {location_choice}", 'font': {'size': 14}},
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

    # Technical Validation
    st.markdown("### 💡 Scaling Pipeline Log (Technical Validation)")
    humidity_idx = list(feature_names).index('Humidity3pm')
    scaled_humidity = scenario_scaled.iloc[0, humidity_idx]
    st.info(
        f"🔧 **Scaler:** Trained `StandardScaler` was used. "
        f"Raw 3pm humidity {humidity_3pm}% → scaled value: **{scaled_humidity:.3f}**"
    )