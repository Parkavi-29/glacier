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

# ------------------- SETUP -------------------
ist = pytz.timezone('Asia/Kolkata')
current_time_ist = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S').upper()
st.set_page_config(layout="wide")

# ------------------- CUSTOM NAVBAR -------------------
st.markdown("""
<style>
.navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    background-color: #ffffffcc;
    padding: 10px 20px;
    border-bottom: 1px solid #ccc;
    font-family: 'Catamaran', sans-serif;
}
.navbar a {
    margin-right: 20px;
    font-size: 18px;
    color: #0b3954;
    text-decoration: none;
    font-weight: 600;
}
.navbar a:hover {
    color: #0077b6;
}
</style>
<div class="navbar">
    <a href="#overview">Overview</a>
    <a href="#charts">Trends</a>
    <a href="#prediction">Forecast</a>
    <a href="#alerts">Alerts</a>
    <a href="#map">Map</a>
</div>
""", unsafe_allow_html=True)

# ------------------- CLOCK -------------------
st.markdown(f"""
<div style="text-align: center; font-family: 'Catamaran', sans-serif; margin-top: 10px; padding-bottom: 10px;">
    <div style="font-size: 24px; color: #333;">{current_time_ist}</div>
</div>
""", unsafe_allow_html=True)

# ------------------- WELCOME -------------------
st.markdown("""
<div style="text-align: center; margin-top: 20px; margin-bottom: 30px;">
    <h2 style="color: #0b3954;">Welcome to the Gangotri Glacier Melt Dashboard</h2>
    <p style="font-size: 18px; max-width: 800px; margin: auto;">
        Explore glacier area trends, make predictions, view interactive maps, and stay alert to climate changes — powered by satellite data and AI.
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------- STYLING -------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Catamaran:wght@400;600;700&display=swap" rel="stylesheet">
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/free-vector/gradient-background-green-tones_23-2148373477.jpg");
    background-color: transparent;
    background-size: cover;
    background-attachment: fixed;
    background-repeat: no-repeat;
    font-family: 'Catamaran', sans-serif;
}
section.main {
    background-color: rgba(255, 255, 255, 0.3);
    padding: 1.5rem;
    border-radius: 10px;
}
h1, h2, h3 {
    color: #0b3954 !important;
    font-weight: 700;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.85);
}
</style>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR -------------------
with st.sidebar:
    show_sidebar = st.toggle("🔽 Show Sidebar", value=True)

if show_sidebar:
    st.sidebar.title("🧨 Glacier Dashboard")
    page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])
else:
    page = "Overview"

# ------------------- CHATBOT -------------------
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

# ------------------- LOAD DATA -------------------
csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Gangotri_Glacier_Area_NDSI_2001_2023.csv'
try:
    df = pd.read_csv(csv_url)
    df = df.dropna(subset=['year', 'area_km2'])
    st.success("✅ Data loaded from GitHub!")
except Exception as e:
    st.error("❌ Failed to load CSV data.")
    st.exception(e)
    df = None

# ------------------- GLOBAL TIME SLIDER -------------------
if df is not None:
    year_min, year_max = int(df['year'].min()), int(df['year'].max())
    year_range = st.slider("📆 Select year range:", year_min, year_max, (year_min, year_max), step=1)
    df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

# ------------------- PAGE LOGIC -------------------
if df is not None:
    if page == "Overview":
        st.markdown("<h2 id='overview'>📋 Gangotri Glacier Melt Overview</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div style="border: 2px solid #0b3954; padding: 20px; border-radius: 10px; background-color: rgba(255, 255, 255, 0.95);">
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
        st.dataframe(df_filtered, use_container_width=True)

    elif page == "Chart View":
        st.markdown("<h2 id='charts'>📈 Glacier Retreat Trends</h2>", unsafe_allow_html=True)
        fig = px.line(df_filtered, x='year', y='area_km2', markers=True, title="Glacier Area (2001–2023)")
        st.plotly_chart(fig, use_container_width=True)
        loss = df_filtered['area_km2'].max() - df_filtered['area_km2'].min()
        st.metric("📉 Total Glacier Loss", f"{loss:.2f} sq.km")

    elif page == "Prediction":
        st.markdown("<h2 id='prediction'>🔮 Glacier Area Forecast</h2>", unsafe_allow_html=True)
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

        for year, value in zip(future_years.flatten(), pred_poly.flatten()):
            st.metric(f"📈 Predicted Area ({year})", f"{value:.2f} sq.km")

        st.subheader("📊 ARIMA Time Series Forecast")
        try:
            model_arima = ARIMA(df_model['area_km2'], order=(1, 1, 1))
            model_fit = model_arima.fit()
            forecast = model_fit.forecast(steps=10)
            future_years_arima = np.arange(df_model['year'].iloc[-1] + 1, df_model['year'].iloc[-1] + 11)
            arima_df = pd.DataFrame({
                'year': future_years_arima,
                'area_km2': forecast,
                'type': 'ARIMA Forecast'
            })

            all_df = pd.concat([df_model[['year', 'area_km2', 'type']], arima_df])
            fig_arima = px.line(all_df, x='year', y='area_km2', color='type', title="ARIMA Forecast")
            st.plotly_chart(fig_arima, use_container_width=True)
        except Exception as e:
            st.warning("⚠️ ARIMA forecast failed.")

    elif page == "Alerts":
        st.markdown("<h2 id='alerts'>🚨 Glacier Risk Alerts</h2>", unsafe_allow_html=True)
        latest_area = df_filtered['area_km2'].iloc[-1]
        threshold = 20.0
        if latest_area < threshold:
            st.error(f"🔴 Critical: Glacier area critically low ({latest_area:.2f} sq.km)")
        elif latest_area < threshold + 5:
            st.warning(f"🟡 Warning: Glacier nearing danger ({latest_area:.2f} sq.km)")
        else:
            st.success(f"🟢 Glacier stable. Current: {latest_area:.2f} sq.km")

    elif page == "Map Overview":
        st.markdown("<h2 id='map'>🗺 Gangotri Glacier Map Overview</h2>", unsafe_allow_html=True)
        m = leafmap.Map(center=[30.96, 79.08], zoom=11)
        m.to_streamlit(height=600)

# ------------------- FOOTER -------------------
st.markdown("""
<hr>
<div style='text-align: center; padding: 10px; font-size: 14px; color: #555;'>
    Built with ❤️ using Streamlit | © 2025 Glacier AI Team
</div>
""", unsafe_allow_html=True)
