# ====================================
# FINAL PRO STREAMLIT CODE (2025)
# Gangotri Glacier Interactive Dashboard
# ====================================

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

# ------------------- SETUP -------------------
ist = pytz.timezone('Asia/Kolkata')
now = datetime.now(ist)
current_time = now.strftime('%I:%M %p')
current_date = now.strftime('%d %B %Y')

st.set_page_config(layout="wide")

# ------------------- STYLING -------------------
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

# ------------------- HEADER -------------------
st.markdown(f"""
<div style="text-align: center; color: #003366; padding: 10px; font-family: 'Catamaran', sans-serif;">
    <div style="font-size: 60px; font-weight: bold;">{current_time}</div>
    <div style="font-size: 28px;">{current_date}</div>
</div>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR NAVIGATION -------------------
st.sidebar.title("🧨 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# ------------------- 3-COLUMN LAYOUT -------------------
col1, col2, col3 = st.columns([2, 1.5, 2])

# ------------------- SIMPLE CHATBOT -------------------
with col2:
    st.header("🤖 GlacierBot (Chat)")
    user_q = st.text_input("Ask about glaciers!", placeholder="e.g., What is NDSI?")
    if user_q:
        q = user_q.lower()
        if "ndsi" in q:
            st.success("🧊 NDSI = Normalized Difference Snow Index (detects snow/ice)")
        elif "gangotri" in q:
            st.success("🗻 Gangotri Glacier is the source of River Ganga!")
        elif "retreat" in q:
            st.success("📉 Retreat = glacier melting backward over time.")
        elif "climate" in q:
            st.success("🌡 Climate change increases glacier melting!")
        else:
            st.info("🤖 Try asking about NDSI, retreat, Gangotri glacier, Landsat...")

# ------------------- IMAGE GALLERY -------------------
with col3:
    st.header("🏔 Gangotri Glacier Views")
    st.image("https://www.remotelands.com/travelogues/app/uploads/2018/06/DSC00976-2.jpg", caption="Gangotri Glacier", use_container_width=True)
    st.image("https://akm-img-a-in.tosshub.com/aajtak/images/story/202209/gaumukh_gangotri_glacier_getty_1-sixteen_nine.jpg?size=948:533", caption="Gaumukh (source)", use_container_width=True)

# ------------------- MAIN CONTENT -------------------
with col1:
    # ------------------- AOI DETAILS -------------------
    with st.expander("📍 Area of Interest (AOI) Info"):
        st.markdown("""
        - **🗺 Total Area Covered:** ~64.13 sq.km  
        - **🔁 Dimensions:**  
          • Longitude: 79.03°E to 79.10°E (~7.7 km)  
          • Latitude: 30.94°N to 31.02°N (~8.9 km)  
        - **📌 Places Included:**  
          - Gangotri Glacier (main glacier body)  
          - Gaumukh Snout (glacier terminus)  
          - Chirbasa & Bhojbasa (popular trek points)  
          - Partial Tapovan Valley  
          - Region inside **Gangotri National Park**
        """)

    # ------------------- LOAD DATA -------------------
    csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Gangotri_Glacier_Area_NDSI_2001_2023.csv'
    try:
        df = pd.read_csv(csv_url)
        df = df.dropna(subset=['year', 'area_km2'])
        st.success("✅ Data loaded successfully!")
    except Exception as e:
        st.error("❌ Failed to load data!")
        st.exception(e)
        df = None

    if df is not None:
        if page == "Overview":
            st.title("📋 Glacier Melt Analysis")
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇️ Download Data", df.to_csv(index=False), "gangotri_glacier.csv", "text/csv")

        elif page == "Chart View":
            st.title("📈 Glacier Retreat Trends")
            fig = px.line(df, x='year', y='area_km2', markers=True, title="Observed Retreat (2001–2023)")
            st.plotly_chart(fig, use_container_width=True)

        elif page == "Prediction":
            st.title("🔮 Glacier Area Forecast")
            df_model = df.copy()
            X = df_model['year'].values.reshape(-1, 1)
            y = df_model['area_km2'].values.reshape(-1, 1)

            forecast_method = st.radio("Choose Forecast Method:", ("Polynomial Regression", "ARIMA"))

            if forecast_method == "Polynomial Regression":
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

            elif forecast_method == "ARIMA":
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
            st.title("🚨 Glacier Risk Alerts")
            latest_area = df['area_km2'].iloc[-1]
            threshold = st.slider("Set Alert Threshold (sq.km)", 5, 30, 20)
            if latest_area < threshold:
                st.error(f"🔴 Critical: Area critically low! ({latest_area:.2f} sq.km)")
            elif latest_area < threshold + 5:
                st.warning(f"🟡 Warning: Area nearing danger! ({latest_area:.2f} sq.km)")
            else:
                st.success(f"🟢 Glacier Stable ({latest_area:.2f} sq.km)")

        elif page == "Map Overview":
            st.title("🗺 Map Overview")
            m = leafmap.Map(center=[30.96, 79.08], zoom=11)
            m.to_streamlit(height=600)
