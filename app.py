import os
import subprocess

# auto train model if not exists
if not os.path.exists("model.pkl"):
    subprocess.run(["python", "src/train_model.py"])


    
import streamlit as st
import joblib
import pandas as pd

# page config
st.set_page_config(page_title="🌍 AQI AI System", layout="wide")

# custom CSS (🔥 futuristic look)
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .main {
        background: linear-gradient(135deg, #0e1117, #1c1f2b);
    }
    h1, h2, h3 {
        text-align: center;
        color: #00f2ff;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00f2ff, #007bff);
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# load model
model = joblib.load("model.pkl")

# title
st.title("🌍 AI-Powered Air Quality Prediction System")

st.markdown("---")

# ---------------- INPUT SECTION ---------------- #

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

st.markdown("---")

if st.button("🚀 Predict AQI", key="predict"):

    data = pd.DataFrame([[pm25, pm10, no, no2, nox, nh3, co, so2, o3,
                          benzene, toluene, xylene, year, month]],
                        columns=['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                                 'CO', 'SO2', 'O3', 'Benzene',
                                 'Toluene', 'Xylene', 'year', 'month'])

    result = model.predict(data)
    aqi = result[0]

    st.markdown("## 🎯 Prediction Result")

    if aqi <= 50:
        st.success(f"🌿 AQI: {aqi:.2f} (Good)")
    elif aqi <= 100:
        st.warning(f"🌤 AQI: {aqi:.2f} (Moderate)")
    elif aqi <= 200:
        st.error(f"😷 AQI: {aqi:.2f} (Poor)")
    else:
        st.error(f"☠️ AQI: {aqi:.2f} (Very Poor)")

    st.markdown("---")

    # Health advice
    st.subheader("🏥 Health Advisory")

    if aqi > 200:
        st.error("🚨 Avoid going outside!")
    elif aqi > 100:
        st.warning("😷 Mask recommended")
    else:
        st.success("😊 Air is safe")

# ---------------- GRAPH ---------------- #

st.markdown("---")

st.subheader("📈 AQI Trend Dashboard")

if st.button("📊 Show Graph", key="graph"):
    df = pd.read_csv("data/cleaned_data.csv")
    st.line_chart(df.tail(100)["AQI"])

# ---------------- FUTURE ---------------- #

st.markdown("---")

st.subheader("🔮 Future AQI Prediction")

col6, col7 = st.columns(2)

with col6:
    future_year = st.number_input("Future Year", value=2025)

with col7:
    future_month = st.number_input("Future Month", value=1)

if st.button("🔮 Predict Future AQI", key="future"):

    df = pd.read_csv("data/cleaned_data.csv")
    avg_values = df.mean()

    future_data = pd.DataFrame([[
        avg_values['PM2.5'],
        avg_values['PM10'],
        avg_values['NO'],
        avg_values['NO2'],
        avg_values['NOx'],
        avg_values['NH3'],
        avg_values['CO'],
        avg_values['SO2'],
        avg_values['O3'],
        avg_values['Benzene'],
        avg_values['Toluene'],
        avg_values['Xylene'],
        future_year,
        future_month
    ]],
    columns=['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
             'CO', 'SO2', 'O3', 'Benzene',
             'Toluene', 'Xylene', 'year', 'month'])

    future_aqi = model.predict(future_data)[0]

    st.success(f"🔮 Future AQI: {future_aqi:.2f}")