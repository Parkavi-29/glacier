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

# ------------------- CLOCK -------------------
st.markdown(f"""
<div style="text-align: center; font-family: 'Catamaran', sans-serif; padding: 10px;">
    <div style="font-size: 36px; font-weight: bold; color: #0b3954;">Current Date & Time (IST)</div>
    <div style="font-size: 26px; color: #333;">{current_time_ist}</div>
</div>
""", unsafe_allow_html=True)

# ------------------- STYLING -------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Catamaran:wght@400;600;700&display=swap" rel="stylesheet">
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.pexels.com/photos/281260/pexels-photo-281260.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500");
    background-size: cover;
    background-attachment: fixed;
    font-family: 'Catamaran', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.75);
}
</style>
""", unsafe_allow_html=True)

# ------------------- NAVIGATION -------------------
st.sidebar.title("🧊 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# ------------------- CHATBOT -------------------
with st.sidebar.expander("💬 Ask GlacierBot"):
    user_q = st.text_input("Ask anything about glaciers:")
    if user_q:
        q = user_q.lower()
        if "ndsi" in q:
            st.write("NDSI = Normalized Difference Snow Index (detects snow and ice).")
        elif "gangotri" in q:
            st.write("Gangotri Glacier is the source of the Ganga River in Uttarakhand.")
        elif "retreat" in q:
            st.write("Glacier retreat means backward melting and reduction in length/area.")
        elif "landsat" in q:
            st.write("This project uses Landsat 5, 7 and 8 imagery from GEE.")
        elif "arima" in q:
            st.write("ARIMA is a time series model used for glacier area forecasting.")
        else:
            st.write("Try asking about: NDSI, Landsat, retreat, GEE, glacier area...")

# ------------------- LOAD DATA -------------------
csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Gangotri_Glacier_Area_NDSI_2001_2023.csv'
try:
    df = pd.read_csv(csv_url)
    df = df.dropna(subset=['year', 'area_km2'])
    year_min, year_max = df['year'].min(), df['year'].max()
    st.success("✅ Data loaded from GitHub")
except Exception as e:
    st.error("❌ Failed to load CSV")
    st.exception(e)
    df = None

# ------------------- TIME SLIDER (GLOBAL) -------------------
if df is not None:
    year_range = st.slider("📅 Select Year Range:", min_value=int(df['year'].min()), max_value=int(df['year'].max()), value=(2001, 2023))
    df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

# ------------------- PAGE ROUTES -------------------
if df is not None:

    if page == "Overview":
        st.title("📋 Gangotri Glacier Melt Overview")
        st.dataframe(df_filtered, use_container_width=True)

    elif page == "Chart View":
        st.title("📈 Glacier Retreat Trend")
        fig = px.line(df_filtered, x='year', y='area_km2', title="Observed Glacier Retreat", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        st.metric("📉 Total Glacier Loss", f"{df_filtered['area_km2'].max() - df_filtered['area_km2'].min():.2f} sq.km")

        st.markdown("""<hr><br>""", unsafe_allow_html=True)
        with st.expander("📍 AOI Summary (Gangotri Region)"):
            st.markdown("""
            - **Total Glacier Area:** ~64.13 sq.km  
            - **Bounding Box:**  
              - Longitude: 79.03°E → 79.10°E (~7.7 km)  
              - Latitude: 30.94°N → 31.02°N (~8.9 km)  
            - **Major Places Covered:**  
              - Gangotri Glacier  
              - Gaumukh Snout  
              - Chirbasa & Bhojbasa  
              - Tapovan (partial)  
              - Gangotri National Park
            """)

    elif page == "Prediction":
        st.title("🔮 Forecast Glacier Area")
        X = df_filtered['year'].values.reshape(-1, 1)
        y = df_filtered['area_km2'].values.reshape(-1, 1)

        st.subheader("📉 Polynomial Regression (up to 2050)")
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)

        future_years = np.arange(2025, 2051, 5).reshape(-1, 1)
        future_poly = poly.transform(future_years)
        pred_poly = model.predict(future_poly)

        df_model = df_filtered.copy()
        df_model['type'] = 'Observed'
        pred_df = pd.DataFrame({'year': future_years.flatten(), 'area_km2': pred_poly.flatten(), 'type': 'Predicted'})
        combined = pd.concat([df_model[['year', 'area_km2', 'type']], pred_df])

        fig = px.line(combined, x='year', y='area_km2', color='type', markers=True, title="Glacier Area Forecast (Polynomial)")
        st.plotly_chart(fig, use_container_width=True)

        for year, val in zip(future_years.flatten(), pred_poly.flatten()):
            st.metric(f"📅 {year}", f"{val:.2f} sq.km")

        st.subheader("📊 ARIMA Forecast (Next 10 Years)")
        try:
            model_arima = ARIMA(df_model['area_km2'], order=(1, 1, 1)).fit()
            forecast = model_arima.forecast(steps=10)
            future_arima_years = np.arange(df_model['year'].iloc[-1] + 1, df_model['year'].iloc[-1] + 11)
            arima_df = pd.DataFrame({'year': future_arima_years, 'area_km2': forecast, 'type': 'ARIMA Forecast'})

            final = pd.concat([df_model[['year', 'area_km2', 'type']], arima_df])
            fig2 = px.line(final, x='year', y='area_km2', color='type', title="ARIMA Forecast")
            st.plotly_chart(fig2, use_container_width=True)
        except:
            st.warning("⚠️ ARIMA forecast failed")

    elif page == "Alerts":
        st.title("🚨 Glacier Risk Alerts")
        current = df_filtered['area_km2'].iloc[-1]
        threshold = st.slider("Set Danger Threshold (sq.km)", 5, 30, 20)

        if current < threshold:
            st.error(f"🔴 Critical Alert: {current:.2f} sq.km")
        elif current < threshold + 5:
            st.warning(f"🟡 Warning: Near threshold! ({current:.2f} sq.km)")
        else:
            st.success(f"🟢 Stable: {current:.2f} sq.km")

    elif page == "Map Overview":
        st.title("🗺 Map Overview")
        m = leafmap.Map(center=[30.96, 79.08], zoom=11)
        m.to_streamlit(height=600)
