import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

# ---------------- PATH FIX ---------------- #
DATA_PATH = "data/cleaned_data.csv"

# ---------------- AUTO TRAIN MODEL ---------------- #
if not os.path.exists("model.pkl"):

    if not os.path.exists(DATA_PATH):
        st.error("❌ Data file not found! Check folder name: data/cleaned_data.csv")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    X = df.drop("AQI", axis=1)
    y = df["AQI"]

    model_temp = RandomForestRegressor(n_estimators=50)
    model_temp.fit(X, y)

    joblib.dump(model_temp, "model.pkl")

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("model.pkl")

# ---------------- UI ---------------- #
st.set_page_config(page_title="🌍 AQI AI System", layout="wide")

st.title("🌍 AI-Powered Air Quality Prediction System")
st.markdown("---")

# ---------------- INPUT ---------------- #
st.subheader("📥 Enter Pollution Data")

col1, col2, col3 = st.columns(3)

with col1:
    pm25 = st.number_input("PM2.5", value=50.0)
    pm10 = st.number_input("PM10", value=80.0)
    no = st.number_input("NO", value=10.0)
    no2 = st.number_input("NO2", value=20.0)

with col2:
    nox = st.number_input("NOx", value=30.0)
    nh3 = st.number_input("NH3", value=5.0)
    co = st.number_input("CO", value=1.0)
    so2 = st.number_input("SO2", value=10.0)

with col3:
    o3 = st.number_input("O3", value=25.0)
    benzene = st.number_input("Benzene", value=2.0)
    toluene = st.number_input("Toluene", value=3.0)
    xylene = st.number_input("Xylene", value=1.0)

st.markdown("---")

col4, col5 = st.columns(2)

with col4:
    year = st.number_input("Year", value=2024)

with col5:
    month = st.number_input("Month", value=1)

# ---------------- PREDICTION ---------------- #
if st.button("🚀 Predict AQI", key="predict"):

    data = pd.DataFrame([[pm25, pm10, no, no2, nox, nh3, co, so2, o3,
                          benzene, toluene, xylene, year, month]],
                        columns=['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                                 'CO', 'SO2', 'O3', 'Benzene',
                                 'Toluene', 'Xylene', 'year', 'month'])

    aqi = model.predict(data)[0]

    st.subheader("🎯 Prediction Result")

    if aqi <= 50:
        st.success(f"🌿 AQI: {aqi:.2f} (Good)")
    elif aqi <= 100:
        st.warning(f"🌤 AQI: {aqi:.2f} (Moderate)")
    elif aqi <= 200:
        st.error(f"😷 AQI: {aqi:.2f} (Poor)")
    else:
        st.error(f"☠️ AQI: {aqi:.2f} (Very Poor)")