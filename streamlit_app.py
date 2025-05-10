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
# ------------------- APP TITLE -------------------
st.markdown("""
<h1 style='text-align: center; color: #0b3954; font-family: Catamaran; font-size: 46px; margin-bottom: 0;'>
    🏔️ Glacier Melt Analysis and Predictions 
</h1>
""", unsafe_allow_html=True)

# ------------------- CLOCK -------------------
st.markdown(f"""
<div style="text-align: center; font-family: 'Catamaran', sans-serif; margin-top: -10px; padding-bottom: 20px;">
    <div style="font-size: 30px; font-weight: bold; color: #0b3954;"></div>
    <div style="font-size: 24px; color: #333;">{current_time_ist}</div>
</div>
""", unsafe_allow_html=True)

# ------------------- STYLING -------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Catamaran:wght@400;600;700&display=swap" rel="stylesheet">
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://media.istockphoto.com/id/1447252624/vector/vector-seamless-geometric-pattern.jpg?s=612x612&w=0&k=20&c=2PaSnkcQcuh5TAXXxb6GfHs2zU2Ah42eUYQg0TNjZII=");
    background-color: transparent;
    background-size: cover;
    background-attachment: fixed;
    background-repeat: no-repeat;
    font-family: 'Catamaran', sans-serif;
}
section.main {
    background-color: rgba(255, 255, 255, 0.3);  /* Light background to improve readability */
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
st.sidebar.title("🧨 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# ------------------- BUILT-IN CHATBOT -------------------
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
        # AOI summary in overview page
        st.markdown("""
        <div style="border: 2px solid #0b3954; padding: 20px; border-radius: 10px; background-color: rgba(255, 255, 255, 0.95); font-family: 'Catamaran', sans-serif;">
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

        st.title("📋 Gangotri Glacier Melt Overview")
        st.dataframe(df_filtered, use_container_width=True)

    elif page == "Chart View":
        st.title("📈 Glacier Retreat Trends")
        fig = px.line(df_filtered, x='year', y='area_km2', markers=True, title="Glacier Area (2001–2023)")
        st.plotly_chart(fig, use_container_width=True)
        loss = df_filtered['area_km2'].max() - df_filtered['area_km2'].min()
        st.metric("📉 Total Glacier Loss", f"{loss:.2f} sq.km")

    elif page == "Prediction":
        st.title("🔮 Glacier Area Forecast")
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
        st.title("🚨 Glacier Risk Alerts")
        latest_area = df_filtered['area_km2'].iloc[-1]
        threshold = 20.0
        if latest_area < threshold:
            st.error(f"🔴 Critical: Glacier area critically low ({latest_area:.2f} sq.km)")
        elif latest_area < threshold + 5:
            st.warning(f"🟡 Warning: Glacier nearing danger ({latest_area:.2f} sq.km)")
        else:
            st.success(f"🟢 Glacier stable. Current: {latest_area:.2f} sq.km")

    elif page == "Map Overview":
        st.title("🗺 Gangotri Glacier Map Overview")
        m = leafmap.Map(center=[30.96, 79.08], zoom=11)
        m.to_streamlit(height=600)
