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

# ---------- TIME AND CONFIG ----------
ist = pytz.timezone('Asia/Kolkata')
current_time_ist = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S').upper()
st.set_page_config(layout="wide")

# ---------- CUSTOM CSS FOR THEME ----------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Catamaran:wght@400;600;700&display=swap" rel="stylesheet">
<style>
body {
    font-family: 'Catamaran', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e0f7fa 0%, #e1f5fe 100%);
}
h1, h2, h3 {
    color: #0b3954;
    font-family: 'Catamaran', sans-serif;
}
.hero {
    text-align: center;
    padding: 2rem 1rem 1.2rem 1rem;
    background: white;
    border-radius: 15px;
    margin-bottom: 25px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}
.hero h1 {
    font-size: 2.7rem;
    margin: 0;
}
.hero p {
    font-size: 1.1rem;
    color: #333;
    margin-top: 0.5rem;
}
.card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.07);
    margin-top: 1rem;
}
[data-testid="stSidebar"] {
    background-color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# ---------- HERO SECTION ----------
st.markdown(f"""
<div class="hero">
    <h1>🏔️ Glacier Melt Analysis and Predictions</h1>
    <p>{current_time_ist}</p>
</div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.title("🧨 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# ---------- SIDEBAR BOT ----------
with st.sidebar.expander("💬 Ask GlacierBot"):
    user_q = st.text_input("Your question:", placeholder="e.g. What is NDSI?")
    if user_q:
        q = user_q.lower()
        if "ndsi" in q:
            st.write("🧊 NDSI = Normalized Difference Snow Index, used to detect snow/ice.")
        elif "gangotri" in q:
            st.write("🗻 Gangotri Glacier is one of the largest Himalayan glaciers and source of the Ganges.")
        elif "retreat" in q:
            st.write("📉 Retreat = melting and shrinkage of glaciers over time.")
        elif "area" in q:
            st.write("🗺 Area is estimated via pixel-based NDSI thresholding (> 0.4).")
        elif "arima" in q:
            st.write("📊 ARIMA is a time-series model for forecasting future glacier area.")
        else:
            st.info("🤖 Try keywords like NDSI, retreat, ARIMA, Gangotri...")

# ---------- LOAD DATA ----------
csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Gangotri_Glacier_Area_NDSI_2001_2023.csv'
try:
    df = pd.read_csv(csv_url)
    df = df.dropna(subset=['year', 'area_km2'])
    st.success("✅ Data loaded from GitHub!")
except Exception as e:
    st.error("❌ Failed to load CSV data.")
    st.exception(e)
    df = None

# ---------- GLOBAL TIME SLIDER ----------
if df is not None:
    year_min, year_max = int(df['year'].min()), int(df['year'].max())
    year_range = st.slider("📆 Select year range:", year_min, year_max, (year_min, year_max), step=1)
    df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

# ---------- PAGE CONTENT ----------
if df is not None:

    if page == "Overview":
        st.markdown("""<div class="card">""", unsafe_allow_html=True)
        st.markdown("""
        ### 📍 Area of Interest (AOI) Summary
        - **Total Glacier Area**: ~64.13 sq.km  
        - **Bounding Box**:
            - Longitude: 79.03°E → 79.10°E (~7.7 km)  
            - Latitude: 30.94°N → 31.02°N (~8.9 km)  
        - **Major Places Covered**:
            - Gangotri Glacier  
            - Gaumukh Snout  
            - Chirbasa & Bhojbasa  
            - Tapovan (partial)  
            - Gangotri National Park
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 📋 Gangotri Glacier Melt Overview")
        st.dataframe(df_filtered, use_container_width=True)

    elif page == "Chart View":
        st.markdown("### 📈 Glacier Retreat Trends")
        fig = px.line(df_filtered, x='year', y='area_km2', markers=True, title="Glacier Area (2001–2023)")
        st.plotly_chart(fig, use_container_width=True)
        loss = df_filtered['area_km2'].max() - df_filtered['area_km2'].min()
        st.metric("📉 Total Glacier Loss", f"{loss:.2f} sq.km")

    elif page == "Prediction":
        st.markdown("### 🔮 Glacier Area Forecast")
        df_model = df_filtered.copy()
        X = df_model['year'].values.reshape(-1, 1)
        y = df_model['area_km2'].values.reshape(-1, 1)

        st.subheader("📉 Polynomial Regression (to 2050)")
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)

        future_years = np.arange(2025, 2051, 5).reshape(-1, 1)
        future_poly = poly.transform(future_years)
        pred_poly = model.predict(future_poly)

        pred_df = pd.DataFrame({'year': future_years.flatten(), 'area_km2': pred_poly.flatten(), 'type': 'Predicted'})
        df_model['type'] = 'Observed'
        full_df = pd.concat([df_model[['year', 'area_km2', 'type']], pred_df])

        fig_poly = px.line(full_df, x='year', y='area_km2', color='type', markers=True, title="Polynomial Forecast")
        st.plotly_chart(fig_poly, use_container_width=True)

        st.subheader("📊 ARIMA Time Series Forecast")
        try:
            model_arima = ARIMA(df_model['area_km2'], order=(1, 1, 1))
            model_fit = model_arima.fit()
            forecast = model_fit.forecast(steps=10)
            future_years_arima = np.arange(df_model['year'].iloc[-1] + 1, df_model['year'].iloc[-1] + 11)
            arima_df = pd.DataFrame({'year': future_years_arima, 'area_km2': forecast, 'type': 'ARIMA Forecast'})

            all_df = pd.concat([df_model[['year', 'area_km2', 'type']], arima_df])
            fig_arima = px.line(all_df, x='year', y='area_km2', color='type', title="ARIMA Forecast")
            st.plotly_chart(fig_arima, use_container_width=True)
        except Exception as e:
            st.warning("⚠️ ARIMA forecast failed.")

    elif page == "Alerts":
        st.markdown("### 🚨 Glacier Risk Alerts")
        latest_area = df_filtered['area_km2'].iloc[-1]
        threshold = 20.0
        if latest_area < threshold:
            st.error(f"🔴 Critical: Glacier area critically low ({latest_area:.2f} sq.km)")
        elif latest_area < threshold + 5:
            st.warning(f"🟡 Warning: Glacier nearing danger ({latest_area:.2f} sq.km)")
        else:
            st.success(f"🟢 Glacier stable. Current: {latest_area:.2f} sq.km")

    elif page == "Map Overview":
        st.markdown("### 🗺 Gangotri Glacier Map Overview")
        m = leafmap.Map(center=[30.96, 79.08], zoom=11)
        m.to_streamlit(height=600)
