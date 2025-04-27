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
<div style="font-size: 24px; font-weight: bold; text-transform: uppercase;">
🕒 Current Date & Time (IST): {current_time_ist}
</div>
""", unsafe_allow_html=True)

# ------------------- STYLING -------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Catamaran:wght@400;600;700&display=swap" rel="stylesheet">
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/premium-vector/blue-mountain-landscape-with-blue-mountain-range-background_530386-237.jpg");
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
    font-family: 'Catamaran', sans-serif;
    font-weight: 700;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.75);
}
</style>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR NAV -------------------
st.sidebar.title("🧨 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# ------------------- SIMPLE BUILT-IN CHATBOT -------------------
with st.sidebar.expander("💬 Ask GlacierBot"):
    st.markdown("I can answer glacier-related questions! Try:")
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
        elif "elevation" in q:
            st.write("📏 Mean elevation gives average height of the glacier zone, useful in melt analysis.")
        elif "arima" in q:
            st.write("📊 ARIMA is a time series forecasting model used for glacier area prediction.")
        elif "regression" in q:
            st.write("📉 Polynomial regression helps model glacier area trends over years.")
        elif "satellite" in q or "landsat" in q:
            st.write("🛰 This app uses Landsat 5, 7, and 8 imagery from Google Earth Engine.")
        elif "climate" in q:
            st.write("🌡 Climate change is a major reason for glacier melt globally.")
        elif "threshold" in q or "alert" in q:
            st.write("⚠️ Alerts are based on a critical glacier area threshold (e.g., < 20 sq.km).")
        else:
            st.write("🤖 I'm still learning. Ask me about NDSI, retreat, Landsat, ARIMA, etc.")

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

# ------------------- PAGES -------------------
if df is not None:
    if page == "Overview":
        st.title("📋 Glacier Melt Analysis (Gangotri)")
        st.markdown("Analyzing Gangotri Glacier retreat from Landsat Data (NDSI-based, 2001-2023)")
        st.dataframe(df, use_container_width=True)

    elif page == "Chart View":
        st.title("📈 Glacier Retreat Trends")
        fig_area = px.line(df, x='year', y='area_km2', markers=True, title="Observed Glacier Retreat")
        st.plotly_chart(fig_area, use_container_width=True)
        st.metric("📉 Total Glacier Loss", f"{df['area_km2'].max() - df['area_km2'].min():.2f} sq.km")

    elif page == "Prediction":
        st.title("🔮 Future Glacier Area Prediction")

        df_model = df.copy()
        X = df_model['year'].values.reshape(-1, 1)
        y = df_model['area_km2'].values.reshape(-1, 1)

        st.subheader("📉 Polynomial Regression (to 2050)")
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)

        future_years = np.arange(2025, 2051, 5).reshape(-1, 1)
        future_poly = poly.transform(future_years)
        pred_poly = model.predict(future_poly)

        pred_df = pd.DataFrame({
            'year': future_years.flatten(),
            'area_km2': pred_poly.flatten(),
            'type': 'Predicted'
        })
        df_model['type'] = 'Observed'
        full_df = pd.concat([df_model[['year', 'area_km2', 'type']], pred_df])

        fig = px.line(full_df, x='year', y='area_km2', color='type', markers=True,
                      title="Glacier Area Forecast (Polynomial Regression)")
        st.plotly_chart(fig, use_container_width=True)

        for year, value in zip(future_years.flatten(), pred_poly.flatten()):
            st.metric(f"📈 Predicted Area ({year})", f"{value:.2f} sq.km")

        st.subheader("📊 ARIMA Time Series Forecast (next 10 years)")
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
            st.warning("⚠️ ARIMA forecast failed. Consider adjusting parameters.")

    elif page == "Alerts":
        st.title("🚨 Glacier Risk Alerts")
        latest_area = df['area_km2'].iloc[-1]
        threshold = 20.0
        if latest_area < threshold:
            st.error(f"🔴 Critical Alert: Glacier area critically low! ({latest_area:.2f} sq.km)")
        elif latest_area < threshold + 5:
            st.warning(f"🟡 Warning: Glacier nearing danger ({latest_area:.2f} sq.km)")
        else:
            st.success(f"🟢 Glacier stable. Current: {latest_area:.2f} sq.km")

    elif page == "Map Overview":
        st.title("🗺 Gangotri Glacier Map Overview")
        m = leafmap.Map(center=[30.96, 79.08], zoom=11)
        m.to_streamlit(height=600)
