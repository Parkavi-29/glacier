# FINAL STREAMLIT APP (2025) - Gangotri Glacier Melt Dashboard

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import leafmap.foliumap as leafmap
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import pytz

# Setup
ist = pytz.timezone('Asia/Kolkata')
now = datetime.now(ist)
current_time = now.strftime('%I:%M %p')
current_date = now.strftime('%d %B %Y')
st.set_page_config(layout="wide")

# ------------------- Styling -------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Catamaran:wght@400;600;700&display=swap" rel="stylesheet">
<style>
[data-testid="stAppViewContainer"] {
    background: url('https://wallpapers.com/images/hd/light-color-background-ktlguo4sk6owzjuh.jpg');
    background-size: cover;
    background-position: center;
    font-family: 'Catamaran', sans-serif;
}
.main {
    background-color: rgba(255, 255, 255, 0.88);
    padding: 2rem;
    border-radius: 10px;
}
h1, h2, h3 {
    color: #003366 !important;
    font-family: 'Catamaran', sans-serif;
    font-weight: 700;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.9);
}
</style>
""", unsafe_allow_html=True)

# ------------------- Header -------------------
st.markdown(f"""
<div style="text-align: center; color: #003366; padding: 10px; font-family: 'Catamaran', sans-serif;">
    <div style="font-size: 60px; font-weight: bold;">{current_time}</div>
    <div style="font-size: 28px;">{current_date}</div>
</div>
""", unsafe_allow_html=True)

# ------------------- Sidebar Navigation -------------------
st.sidebar.title("🧨 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# ------------------- 3-Column Layout -------------------
col1, col2, col3 = st.columns([2, 1.5, 2])

# ------------------- Chatbot -------------------
with col2:
    st.header("🤖 GlacierBot")
    q = st.text_input("Ask about glaciers!", placeholder="e.g. What is NDSI?")
    if q:
        query = q.lower()
        if "ndsi" in query:
            st.success("🧊 NDSI = Normalized Difference Snow Index (detects snow/ice)")
        elif "gangotri" in query:
            st.success("🗻 Gangotri Glacier is the source of River Ganga!")
        elif "retreat" in query:
            st.success("📉 Retreat = glacier melting backward over time.")
        elif "climate" in query:
            st.success("🌡 Climate change increases glacier melting!")
        else:
            st.info("🤖 Try asking about NDSI, retreat, Gangotri glacier, Landsat...")

# ------------------- Image Gallery -------------------
with col3:
    st.header("🏔 Gangotri Views")
    st.image("https://www.remotelands.com/travelogues/app/uploads/2018/06/DSC00976-2.jpg", caption="Gangotri Glacier", use_container_width=True)
    st.image("https://akm-img-a-in.tosshub.com/aajtak/images/story/202209/gaumukh_gangotri_glacier_getty_1-sixteen_nine.jpg?size=948:533", caption="Gaumukh Snout", use_container_width=True)

# ------------------- Main Content -------------------
with col1:
    # AOI Summary
    with st.expander("📍 Area of Interest (AOI)"):
        st.markdown("""
        - **🧊 Total Glacier Area:** ~64.13 sq.km  
        - **Longitude:** 79.03°E to 79.10°E (~7.7 km)  
        - **Latitude:** 30.94°N to 31.02°N (~8.9 km)  
        - **Covered Areas:**  
            • Gangotri Glacier  
            • Gaumukh Snout  
            • Chirbasa & Bhojbasa  
            • Partial Tapovan Valley  
            • Gangotri National Park  
        """)

    # Load Data
    csv_url = "https://raw.githubusercontent.com/Parkavi-29/glacier/main/Gangotri_Glacier_Area_NDSI_2001_2023.csv"
    try:
        df = pd.read_csv(csv_url).dropna(subset=['year', 'area_km2'])
        st.success("✅ Data loaded!")
    except Exception as e:
        st.error("❌ Data load failed.")
        st.exception(e)
        df = None

    # Pages
    if df is not None:
        if page == "Overview":
            st.title("📋 Glacier Melt Analysis")
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇ Download Data", df.to_csv(index=False), "glacier_data.csv", "text/csv")

        elif page == "Chart View":
            st.title("📈 Glacier Retreat Trends (2001–2023)")
            fig_obs = px.line(df, x='year', y='area_km2', markers=True, title="Observed Glacier Retreat")
            st.plotly_chart(fig_obs, use_container_width=True)

            # Forecast overlay
            poly = PolynomialFeatures(degree=2)
            X = df['year'].values.reshape(-1, 1)
            y = df['area_km2'].values
            X_poly = poly.fit_transform(X)
            model = LinearRegression().fit(X_poly, y)

            future_years = np.arange(2025, 2036, 1).reshape(-1, 1)
            future_poly = poly.transform(future_years)
            pred = model.predict(future_poly)

            pred_df = pd.DataFrame({'year': future_years.flatten(), 'area_km2': pred, 'type': 'Predicted'})
            obs_df = df.copy()
            obs_df['type'] = 'Observed'
            full_df = pd.concat([obs_df[['year', 'area_km2', 'type']], pred_df])

            fig_forecast = px.line(full_df, x='year', y='area_km2', color='type', markers=True, title="Observed + Forecast Overlay")
            st.plotly_chart(fig_forecast, use_container_width=True)

        elif page == "Prediction":
            st.title("🔮 Forecast: Glacier Area (2025–2050)")
            X = df['year'].values.reshape(-1, 1)
            y = df['area_km2'].values

            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            model = LinearRegression().fit(X_poly, y)

            future_years = np.arange(2025, 2051, 5).reshape(-1, 1)
            future_poly = poly.transform(future_years)
            pred = model.predict(future_poly)

            pred_df = pd.DataFrame({'year': future_years.flatten(), 'area_km2': pred, 'type': 'Predicted'})
            obs_df = df.copy()
            obs_df['type'] = 'Observed'
            full_df = pd.concat([obs_df[['year', 'area_km2', 'type']], pred_df])

            fig_pred = px.line(full_df, x='year', y='area_km2', color='type', markers=True,
                               title="Polynomial Forecast (2025–2050)")
            st.plotly_chart(fig_pred, use_container_width=True)

            # Display predicted loss
            observed_max = df['area_km2'].max()
            predicted_2050 = pred_df[pred_df['year'] == 2050]['area_km2'].values[0]
            loss = observed_max - predicted_2050
            st.metric("📉 Predicted Area Loss (by 2050)", f"{loss:.2f} sq.km")

            # ARIMA Forecast
            st.subheader("📊 ARIMA Forecast (Next 10 years)")
            try:
                model_arima = ARIMA(df['area_km2'], order=(1, 1, 1))
                model_fit = model_arima.fit()
                arima_forecast = model_fit.forecast(steps=10)
                future_years_arima = np.arange(df['year'].iloc[-1] + 1, df['year'].iloc[-1] + 11)
                arima_df = pd.DataFrame({'year': future_years_arima, 'area_km2': arima_forecast, 'type': 'ARIMA'})

                full_arima = pd.concat([obs_df[['year', 'area_km2', 'type']], arima_df])
                fig_arima = px.line(full_arima, x='year', y='area_km2', color='type', markers=True,
                                    title="ARIMA Time Series Forecast")
                st.plotly_chart(fig_arima, use_container_width=True)
            except Exception as e:
                st.warning("⚠️ ARIMA forecast failed.")

        elif page == "Alerts":
            st.title("🚨 Glacier Risk Alert")
            latest_area = df['area_km2'].iloc[-1]
            threshold = st.slider("Alert Threshold (sq.km)", 5, 30, 20)
            if latest_area < threshold:
                st.error(f"🔴 CRITICAL! Area = {latest_area:.2f} sq.km")
            elif latest_area < threshold + 5:
                st.warning(f"🟡 Caution! Area = {latest_area:.2f} sq.km")
            else:
                st.success(f"🟢 Stable: Area = {latest_area:.2f} sq.km")

        elif page == "Map Overview":
            st.title("🗺 Interactive Map")
            m = leafmap.Map(center=[30.96, 79.08], zoom=11)
            m.to_streamlit(height=600)
