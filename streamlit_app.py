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
.main {
    background-color: rgba(255, 255, 255, 0.88);
    padding: 2rem;
    border-radius: 10px;
}
h1, h2, h3 {
    color: #0b3954 !important;
    font-weight: 700;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.75);
}
</style>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR NAV -------------------
st.sidebar.title("🧊 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# ------------------- BUILT-IN CHATBOT -------------------
with st.sidebar.expander("💬 Ask GlacierBot"):
    user_q = st.text_input("Ask about glaciers:", placeholder="e.g. What is NDSI?")
    if user_q:
        q = user_q.lower()
        if "ndsi" in q:
            st.write("🧊 NDSI stands for Normalized Difference Snow Index, used to detect snow and ice.")
        elif "gangotri" in q:
            st.write("🗻 Gangotri Glacier is one of the largest in the Himalayas.")
        elif "retreat" in q:
            st.write("📉 Retreat means glacier is shrinking over time.")
        elif "arima" in q:
            st.write("📊 ARIMA is used to forecast glacier area.")
        elif "regression" in q:
            st.write("📉 Polynomial Regression fits a curve to the area trend.")
        else:
            st.write("🤖 Ask about NDSI, Gangotri, retreat, ARIMA, regression...")

# ------------------- LOAD DATA -------------------
csv_url = "https://raw.githubusercontent.com/Parkavi-29/glacier/main/Gangotri_Glacier_Area_NDSI_2001_2023.csv"
try:
    df = pd.read_csv(csv_url)
    df = df.dropna(subset=['year', 'area_km2'])
    st.success("✅ Glacier data loaded!")
except Exception as e:
    st.error("❌ Failed to load glacier data.")
    st.exception(e)
    df = None

# ------------------- TIME SLIDER -------------------
if df is not None:
    year_range = st.slider("📆 Select Year Range", int(df['year'].min()), int(df['year'].max()), (2001, 2023))
    df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

    if page == "Overview":
        st.title("📋 Glacier Melt Analysis")
        st.markdown("Overview of Gangotri Glacier melt from 2001–2023 (NDSI-based area)")
        st.dataframe(df_filtered, use_container_width=True)

    elif page == "Chart View":
        st.title("📈 Glacier Retreat Trends")
        fig = px.line(df_filtered, x='year', y='area_km2', markers=True, title="Observed Glacier Retreat")
        st.plotly_chart(fig, use_container_width=True)
        st.metric("📉 Total Glacier Loss", f"{df_filtered['area_km2'].max() - df_filtered['area_km2'].min():.2f} sq.km")

        # AOI Summary section under Chart View
        st.markdown("""
        ### 📍 Area of Interest (AOI) Summary
        - **🧊 Total Glacier Area:** ~64.13 sq.km
        - **📌 Bounding Box:**
            - Longitude: 79.03°E → 79.10°E (~7.7 km)
            - Latitude: 30.94°N → 31.02°N (~8.9 km)
        - **🗺 Major Locations Covered:**
            - Gangotri Glacier  
            - Gaumukh Snout  
            - Chirbasa & Bhojbasa  
            - Tapovan (partial)  
            - Gangotri National Park
        """)

    elif page == "Prediction":
        st.title("🔮 Glacier Area Forecast")
        df_model = df_filtered.copy()
        X = df_model['year'].values.reshape(-1, 1)
        y = df_model['area_km2'].values.reshape(-1, 1)

        st.subheader("📉 Polynomial Regression Forecast")
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)

        future_years = np.arange(2025, 2051, 5).reshape(-1, 1)
        future_poly = poly.transform(future_years)
        pred_poly = model.predict(future_poly)

        pred_df = pd.DataFrame({'year': future_years.flatten(), 'area_km2': pred_poly.flatten(), 'type': 'Predicted'})
        df_model['type'] = 'Observed'
        full_df = pd.concat([df_model[['year', 'area_km2', 'type']], pred_df])

        fig = px.line(full_df, x='year', y='area_km2', color='type', markers=True,
                      title="Polynomial Regression Forecast (to 2050)")
        st.plotly_chart(fig, use_container_width=True)

        for year, value in zip(future_years.flatten(), pred_poly.flatten()):
            st.metric(f"📈 Predicted Area ({year})", f"{value:.2f} sq.km")

        st.subheader("📊 ARIMA Forecast (next 10 years)")
        try:
            model_arima = ARIMA(df_model['area_km2'], order=(1, 1, 1))
            model_fit = model_arima.fit()
            forecast = model_fit.forecast(steps=10)
            future_years_arima = np.arange(df_model['year'].iloc[-1] + 1, df_model['year'].iloc[-1] + 11)
            arima_df = pd.DataFrame({'year': future_years_arima, 'area_km2': forecast, 'type': 'ARIMA Forecast'})
            all_df = pd.concat([df_model[['year', 'area_km2', 'type']], arima_df])

            fig_arima = px.line(all_df, x='year', y='area_km2', color='type', title="ARIMA Forecast - Glacier Area")
            st.plotly_chart(fig_arima, use_container_width=True)
        except Exception as e:
            st.warning("⚠️ ARIMA forecast failed.")
            st.exception(e)

    elif page == "Alerts":
        st.title("🚨 Glacier Risk Alerts")
        latest_area = df_filtered['area_km2'].iloc[-1]
        threshold = 20.0
        if latest_area < threshold:
            st.error(f"🔴 CRITICAL: Area dangerously low! ({latest_area:.2f} sq.km)")
        elif latest_area < threshold + 5:
            st.warning(f"🟡 Warning: Glacier area approaching threshold ({latest_area:.2f} sq.km)")
        else:
            st.success(f"🟢 Glacier area is safe ({latest_area:.2f} sq.km)")

    elif page == "Map Overview":
        st.title("🗺 Gangotri Glacier Map")
        m = leafmap.Map(center=[30.96, 79.08], zoom=11)
        m.to_streamlit(height=600)
