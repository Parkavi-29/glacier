import streamlit as st
import pandas as pd
import plotly.express as px
import leafmap.foliumap as leafmap
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import pytz

# ---------------- SETUP ----------------
ist = pytz.timezone('Asia/Kolkata')
current_time_ist = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S').upper()
st.set_page_config(layout="wide")

# ---------------- HEADER: CLOCK ----------------
st.markdown(f"""
<div style="text-align: center; font-family: 'Catamaran', sans-serif; padding: 10px;">
    <div style="font-size: 36px; font-weight: bold; color: #0b3954;">Current Date & Time (IST)</div>
    <div style="font-size: 26px; color: #333;">{current_time_ist}</div>
</div>
""", unsafe_allow_html=True)

# ---------------- AOI INFO ----------------
st.markdown("""
<div style="border: 2px solid #0b3954; padding: 20px; border-radius: 10px; background-color: rgba(255, 255, 255, 0.93); font-family: 'Catamaran', sans-serif;">
    <h3 style="color: #0b3954;">📍 Area of Interest (AOI) Summary</h3>
    <ul style="line-height: 1.7; font-size: 16px;">
        <li><b>Total Glacier Area:</b> ~64.13 sq.km</li>
        <li><b>Bounding Box:</b>
            <ul>
                <li>Longitude: 79.03°E → 79.10°E (~7.7 km)</li>
                <li>Latitude: 30.94°N → 31.02°N (~8.9 km)</li>
            </ul>
        </li>
        <li><b>Major Places Covered:</b>
            <ul>
                <li>Gangotri Glacier</li>
                <li>Gaumukh Snout</li>
                <li>Chirbasa & Bhojbasa</li>
                <li>Tapovan (partial)</li>
                <li>Gangotri National Park</li>
            </ul>
        </li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ---------------- STYLING ----------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Catamaran:wght@400;600;700&display=swap" rel="stylesheet">
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.pexels.com/photos/281260/pexels-photo-281260.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500");
    background-size: cover;
    background-attachment: fixed;
    font-family: 'Catamaran', sans-serif;
}
.main {
    background-color: rgba(255, 255, 255, 0.88);
    padding: 2rem;
    border-radius: 10px;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.75);
}
h1, h2, h3 {
    color: #0b3954 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# ---------------- CHATBOT ----------------
with st.sidebar.expander("💬 Ask GlacierBot"):
    user_q = st.text_input("Your question:", placeholder="e.g. What is NDSI?")
    if user_q:
        q = user_q.lower()
        if "ndsi" in q:
            st.write("🧊 NDSI stands for Normalized Difference Snow Index, used to detect snow and ice in satellite images.")
        elif "gangotri" in q:
            st.write("🗻 The Gangotri Glacier is one of the largest glaciers in the Himalayas and source of the Ganges.")
        elif "retreat" in q:
            st.write("📉 Glacier retreat refers to the shrinking of glaciers due to melting over time.")
        elif "area" in q:
            st.write("🗺 Area is calculated by detecting glacier pixels using NDSI threshold > 0.4.")
        elif "arima" in q:
            st.write("📊 ARIMA is a time series forecasting model used for glacier area prediction.")
        elif "regression" in q:
            st.write("📈 Polynomial regression helps model glacier area trends over time.")
        else:
            st.write("🤖 Ask me about NDSI, Landsat, ARIMA, regression, etc.")

# ---------------- LOAD DATA ----------------
csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Gangotri_Glacier_Area_NDSI_2001_2023.csv'
try:
    df = pd.read_csv(csv_url)
    df = df.dropna(subset=['year', 'area_km2'])
    st.success("✅ Data loaded from GitHub!")
except Exception as e:
    st.error("❌ Failed to load CSV data.")
    st.exception(e)
    df = None

# ---------------- GLOBAL TIME SLIDER ----------------
if df is not None:
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())

    st.markdown("### 🎚️ Select Year Range for Analysis")
    year_range = st.slider("Time range:", min_year, max_year, (min_year, max_year), step=1)
    df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

    # ---------------- PAGE VIEWS ----------------
    if page == "Overview":
        st.title("📋 Glacier Melt Analysis (Gangotri)")
        st.markdown(f"Showing NDSI-based area from {year_range[0]} to {year_range[1]}")
        st.dataframe(df_filtered, use_container_width=True)

    elif page == "Chart View":
        st.title("📈 Glacier Retreat Trends")
        fig = px.line(df_filtered, x='year', y='area_km2', markers=True, title="Observed Retreat")
        st.plotly_chart(fig, use_container_width=True)
        loss = df_filtered['area_km2'].max() - df_filtered['area_km2'].min()
        st.metric("📉 Total Glacier Loss", f"{loss:.2f} sq.km")

    elif page == "Prediction":
        st.title("🔮 Future Glacier Area Prediction")
        X = df_filtered['year'].values.reshape(-1, 1)
        y = df_filtered['area_km2'].values.reshape(-1, 1)

        st.subheader("📉 Polynomial Regression (to 2050)")
        poly = PolynomialFeatures(degree=2)
        model = LinearRegression().fit(poly.fit_transform(X), y)

        future_years = np.arange(2025, 2051, 5).reshape(-1, 1)
        pred = model.predict(poly.transform(future_years))

        pred_df = pd.DataFrame({'year': future_years.flatten(), 'area_km2': pred.flatten(), 'type': 'Predicted'})
        df_filtered['type'] = 'Observed'
        full_df = pd.concat([df_filtered[['year', 'area_km2', 'type']], pred_df])

        fig = px.line(full_df, x='year', y='area_km2', color='type', markers=True, title="Polynomial Forecast")
        st.plotly_chart(fig, use_container_width=True)

        for y, val in zip(future_years.flatten(), pred.flatten()):
            st.metric(f"📅 {y}", f"{val:.2f} sq.km")

        st.subheader("📊 ARIMA Forecast (Next 10 Years)")
        try:
            model_arima = ARIMA(df_filtered['area_km2'], order=(1, 1, 1)).fit()
            forecast = model_arima.forecast(10)
            future_years_arima = np.arange(df_filtered['year'].iloc[-1] + 1, df_filtered['year'].iloc[-1] + 11)
            arima_df = pd.DataFrame({'year': future_years_arima, 'area_km2': forecast, 'type': 'ARIMA Forecast'})
            all_df = pd.concat([df_filtered[['year', 'area_km2', 'type']], arima_df])
            fig2 = px.line(all_df, x='year', y='area_km2', color='type', title="ARIMA Forecast")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning("⚠️ ARIMA forecast failed.")

    elif page == "Alerts":
        st.title("🚨 Glacier Risk Alerts")
        latest_area = df_filtered['area_km2'].iloc[-1]
        threshold = 20
        if latest_area < threshold:
            st.error(f"🔴 Critical! Area: {latest_area:.2f} sq.km")
        elif latest_area < threshold + 5:
            st.warning(f"🟡 Warning! Area: {latest_area:.2f} sq.km")
        else:
            st.success(f"🟢 Stable. Area: {latest_area:.2f} sq.km")

    elif page == "Map Overview":
        st.title("🗺 Map Overview (Gangotri Glacier)")
        m = leafmap.Map(center=[30.96, 79.08], zoom=11)
        m.to_streamlit(height=600)
