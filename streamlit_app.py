import streamlit as st
import pandas as pd
import plotly.express as px
import leafmap.foliumap as leafmap
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import pytz
from io import BytesIO

# ------------------- SETUP -------------------
ist = pytz.timezone('Asia/Kolkata')
current_time_ist = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S').upper()
st.set_page_config(layout="wide")

# ------------------- TITLE & TIME -------------------
st.markdown("""
<h1 style='text-align: center; color: #0b3954; font-family: Catamaran; font-size: 46px; margin-bottom: 0;'>🏔️ Glacier Melt Analysis and Predictions</h1>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align: center; font-family: 'Catamaran', sans-serif; margin-top: -10px; padding-bottom: 20px;">
    <div style="font-size: 24px; color: #333;">{current_time_ist}</div>
</div>
""", unsafe_allow_html=True)

# ------------------- STYLING -------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Catamaran:wght@400;600;700&display=swap" rel="stylesheet">
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/free-vector/gradient-background-green-tones_23-2148373477.jpg");
    background-size: cover;
    background-attachment: fixed;
}
section.main {
    background-color: rgba(255, 255, 255, 0.3);
    padding: 1.5rem;
    border-radius: 10px;
}
h1, h2, h3 {
    color: #0b3954 !important;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.85);
}
</style>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR -------------------
st.sidebar.title("🧨 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# ------------------- CHATBOT -------------------
with st.sidebar.expander("💬 Ask GlacierBot"):
    user_q = st.text_input("Your question:", placeholder="e.g. What is NDSI?")
    if user_q:
        q = user_q.lower()
        if "ndsi" in q:
            st.write("🧊 NDSI = Normalized Difference Snow Index, used to detect snow/ice.")
        elif "gangotri" in q:
            st.write("🗻 Gangotri Glacier is a large Himalayan glacier, source of the Ganges.")
        elif "retreat" in q:
            st.write("📉 Retreat = melting and shrinking of glaciers.")
        elif "arima" in q:
            st.write("📊 ARIMA is a time-series forecasting model.")
        else:
            st.info("🤖 Try keywords like NDSI, ARIMA, Gangotri...")

# ------------------- DATA LOAD -------------------
csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Gangotri_Glacier_Area_NDSI_2001_2023.csv'
try:
    df = pd.read_csv(csv_url).dropna(subset=['year', 'area_km2'])
    st.success("✅ Data loaded from GitHub")
except Exception as e:
    st.error("❌ Failed to load data.")
    st.stop()

# ------------------- SLIDER -------------------
year_min, year_max = int(df['year'].min()), int(df['year'].max())
year_range = st.slider("📆 Select year range:", year_min, year_max, (year_min, year_max))
df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

# ------------------- PAGES -------------------
if page == "Overview":
    st.subheader("📍 Area of Interest (AOI): Gangotri Glacier")
    st.markdown("""
    - **Total Area**: ~64.13 sq.km  
    - **Region**: Gaumukh, Bhojbasa, Tapovan  
    - **Coordinates**: 30.94°N–31.02°N, 79.03°E–79.10°E
    """)
    st.dataframe(df_filtered, use_container_width=True)

elif page == "Chart View":
    st.subheader("📉 Glacier Retreat Over Years")
    fig = px.line(df_filtered, x='year', y='area_km2', markers=True, title="Glacier Area (2001–2023)")
    st.plotly_chart(fig, use_container_width=True)
    loss = df_filtered['area_km2'].max() - df_filtered['area_km2'].min()
    st.metric("Total Area Loss", f"{loss:.2f} sq.km")

elif page == "Prediction":
    st.title("🔮 Forecasting Glacier Area")
    forecast_year = st.selectbox("Forecast till year:", options=[2030, 2035, 2040, 2045, 2050])
    tabs = st.tabs(["📈 Polynomial Regression", "📊 ARIMA Forecast"])

    # Prepare data
    df_model = df_filtered.copy()
    X = df_model['year'].values.reshape(-1, 1)
    y = df_model['area_km2'].values.reshape(-1, 1)

    # --- Polynomial ---
    with tabs[0]:
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        r2 = r2_score(y, model.predict(X_poly))
        future_years = np.arange(df_model['year'].max() + 1, forecast_year + 1, 1).reshape(-1, 1)
        future_poly = poly.transform(future_years)
        pred_poly = model.predict(future_poly)

        pred_df = pd.DataFrame({'year': future_years.flatten(), 'area_km2': pred_poly.flatten(), 'type': 'Predicted'})
        df_model['type'] = 'Observed'
        all_df = pd.concat([df_model[['year', 'area_km2', 'type']], pred_df])
        fig = px.line(all_df, x='year', y='area_km2', color='type', markers=True, title="Polynomial Forecast")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**R² Score**: {r2:.4f}")
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Forecast CSV", data=csv, file_name="poly_forecast.csv")

    # --- ARIMA ---
    with tabs[1]:
        try:
            arima_model = ARIMA(df_model['area_km2'], order=(1, 1, 1))
            arima_fit = arima_model.fit()
            steps = forecast_year - df_model['year'].max()
            arima_forecast = arima_fit.forecast(steps=steps)
            arima_years = np.arange(df_model['year'].max() + 1, forecast_year + 1)
            arima_df = pd.DataFrame({'year': arima_years, 'area_km2': arima_forecast, 'type': 'ARIMA Forecast'})
            full_df = pd.concat([df_model[['year', 'area_km2', 'type']], arima_df])
            fig2 = px.line(full_df, x='year', y='area_km2', color='type', title="ARIMA Forecast")
            st.plotly_chart(fig2, use_container_width=True)
            csv2 = arima_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download ARIMA Forecast CSV", data=csv2, file_name="arima_forecast.csv")
        except Exception as e:
            st.warning("ARIMA forecast failed.")

elif page == "Alerts":
    st.subheader("🚨 Glacier Risk Alerts")
    latest_area = df_filtered['area_km2'].iloc[-1]
    if latest_area < 20:
        st.error(f"🔴 Critical: Area very low ({latest_area:.2f} sq.km)")
    elif latest_area < 25:
        st.warning(f"🟡 Warning: Area declining ({latest_area:.2f} sq.km)")
    else:
        st.success(f"🟢 Glacier stable: {latest_area:.2f} sq.km")

elif page == "Map Overview":
    st.subheader("🗺️ Interactive Map")
    m = leafmap.Map(center=[30.96, 79.08], zoom=11)
    m.to_streamlit(height=600)
