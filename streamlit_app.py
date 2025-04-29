# Streamlit Glacier Melt App with AOI Page and Time Slider in Chart View

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

# ------------------- SIDEBAR NAVIGATION -------------------
st.sidebar.title("🧨 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["AOI Info", "Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# ------------------- CHATBOT -------------------
with st.sidebar.expander("💬 Ask GlacierBot"):
    st.markdown("Try asking: What is NDSI?")
    user_q = st.text_input("Your question:")
    if user_q:
        q = user_q.lower()
        responses = {
            "ndsi": "🧊 NDSI is Normalized Difference Snow Index, used to detect snow and ice.",
            "gangotri": "🗻 Gangotri Glacier is the source of the Ganges River.",
            "retreat": "📉 Retreat means glacier shrinking due to melting.",
            "arima": "📊 ARIMA is a time series model for forecasting future glacier area.",
            "regression": "📉 Polynomial regression fits curves to past area data.",
            "climate": "🌡 Climate change accelerates glacier melt globally."
        }
        answered = False
        for key in responses:
            if key in q:
                st.write(responses[key])
                answered = True
                break
        if not answered:
            st.info("🤖 I'm still learning! Ask about NDSI, retreat, Landsat...")

# ------------------- DATA LOADING -------------------
csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Gangotri_Glacier_Area_NDSI_2001_2023.csv'
try:
    df = pd.read_csv(csv_url)
    df = df.dropna(subset=['year', 'area_km2'])
    df['year'] = df['year'].astype(int)
except Exception as e:
    st.error("❌ Failed to load data.")
    st.stop()

# ------------------- AOI PAGE -------------------
if page == "AOI Info":
    st.title("📍 Area of Interest (AOI) Summary")
    st.markdown("""
    <div style="border: 2px solid #0b3954; padding: 20px; border-radius: 10px; background-color: rgba(255, 255, 255, 0.93); font-family: 'Catamaran', sans-serif;">
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

# ------------------- OVERVIEW -------------------
elif page == "Overview":
    st.title("📋 Glacier Melt Analysis (Gangotri)")
    st.dataframe(df, use_container_width=True)

# ------------------- CHART VIEW -------------------
elif page == "Chart View":
    st.title("📈 Glacier Retreat Trends")
    min_year, max_year = int(df['year'].min()), int(df['year'].max())
    year_range = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))
    df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    fig = px.line(df_filtered, x='year', y='area_km2', markers=True, title=f"Retreat Trend: {year_range[0]}–{year_range[1]}")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("📉 Total Glacier Loss", f"{df_filtered['area_km2'].max() - df_filtered['area_km2'].min():.2f} sq.km")

# ------------------- PREDICTION -------------------
elif page == "Prediction":
    st.title("🔮 Future Glacier Area Prediction")
    df_model = df.copy()
    X = df_model['year'].values.reshape(-1, 1)
    y = df_model['area_km2'].values.reshape(-1, 1)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)

    future_years = np.arange(2025, 2051, 5).reshape(-1, 1)
    future_poly = poly.transform(future_years)
    pred_poly = model.predict(future_poly)

    pred_df = pd.DataFrame({'year': future_years.flatten(), 'area_km2': pred_poly.flatten(), 'type': 'Predicted'})
    df_model['type'] = 'Observed'
    full_df = pd.concat([df_model[['year', 'area_km2', 'type']], pred_df])

    st.subheader("📉 Polynomial Regression")
    st.plotly_chart(px.line(full_df, x='year', y='area_km2', color='type', markers=True), use_container_width=True)

    for year, value in zip(future_years.flatten(), pred_poly.flatten()):
        st.metric(f"📈 Predicted Area ({year})", f"{value:.2f} sq.km")

    st.subheader("📊 ARIMA Forecast")
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

# ------------------- ALERTS -------------------
elif page == "Alerts":
    st.title("🚨 Glacier Risk Alerts")
    current_area = df['area_km2'].iloc[-1]
    threshold = st.slider("Set Alert Threshold", 10, 30, 20)
    if current_area < threshold:
        st.error(f"🔴 Critical: {current_area:.2f} sq.km — Below threshold!")
    elif current_area < threshold + 5:
        st.warning(f"🟡 Warning: {current_area:.2f} sq.km — Close to danger")
    else:
        st.success(f"🟢 Stable: {current_area:.2f} sq.km")

# ------------------- MAP -------------------
elif page == "Map Overview":
    st.title("🗺 Map Overview")
    m = leafmap.Map(center=[30.96, 79.08], zoom=11)
    m.to_streamlit(height=600)
