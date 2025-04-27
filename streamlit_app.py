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
import time

# ------------------- PAGE SETUP -------------------
st.set_page_config(page_title="Gangotri Glacier Melt Analysis", page_icon="🏊", layout="wide")

# ------------------- CUSTOM LOADING -------------------
with st.spinner('Loading Glacier Dashboard...'):
    time.sleep(1.5)

# ------------------- CLOCK -------------------
ist = pytz.timezone('Asia/Kolkata')
now = datetime.now(ist)
current_time = now.strftime('%-I:%M')
current_date = now.strftime('%a, %d %B')

st.markdown(f"""
<div style="text-align: center; color: white; padding: 10px; font-family: 'Poppins', sans-serif;">
    <div style="font-size: 70px; font-weight: bold;">{current_time}</div>
    <div style="font-size: 30px;">{current_date}</div>
</div>
""", unsafe_allow_html=True)

# ------------------- STYLING -------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
<style>
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(120deg, #f0f2f5 0%, #c9d6ff 100%);
    background-size: cover;
    background-attachment: fixed;
    font-family: 'Poppins', sans-serif;
}
.main {
    background-color: rgba(255, 255, 255, 0.92);
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
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

# ------------------- NAVIGATION -------------------
st.sidebar.title("💨 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# ------------------- CHATBOT -------------------
with st.sidebar.expander("💬 Ask GlacierBot"):
    user_q = st.text_input("Your question:", placeholder="e.g. What is NDSI?")
    if user_q:
        q = user_q.lower()
        if "ndsi" in q:
            st.success("NDSI is Normalized Difference Snow Index for ice detection.")
        elif "gangotri" in q:
            st.success("Gangotri Glacier is a major glacier in Uttarakhand, India.")
        elif "retreat" in q:
            st.success("Retreat means shrinking of glacier area over time.")
        elif "area" in q:
            st.success("Area is computed based on NDSI threshold > 0.4.")
        elif "arima" in q:
            st.success("ARIMA forecasts glacier area into future years.")
        elif "regression" in q:
            st.success("Regression fits glacier area vs years into a curve.")
        else:
            st.info("Try asking about: NDSI, Gangotri, Retreat, ARIMA, Regression!")

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
        st.dataframe(df, use_container_width=True)

    elif page == "Chart View":
        st.title("📈 Glacier Retreat Trends")
        fig = px.line(df, x='year', y='area_km2', markers=True, title="Observed Glacier Retreat")
        st.plotly_chart(fig, use_container_width=True)
        total_loss = df['area_km2'].max() - df['area_km2'].min()
        st.metric("📉 Total Glacier Loss", f"{total_loss:.2f} sq.km")

    elif page == "Prediction":
        st.title("🔮 Future Glacier Area Prediction")
        df_model = df.copy()
        X = df_model['year'].values.reshape(-1, 1)
        y = df_model['area_km2'].values.reshape(-1, 1)

        st.subheader("📉 Polynomial Regression (till 2050)")
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)

        future_years = np.arange(2025, 2051, 5).reshape(-1, 1)
        pred_poly = model.predict(poly.transform(future_years))

        pred_df = pd.DataFrame({
            'year': future_years.flatten(),
            'area_km2': pred_poly.flatten(),
            'type': 'Predicted'
        })
        df_model['type'] = 'Observed'

        full_df = pd.concat([df_model[['year', 'area_km2', 'type']], pred_df])

        fig2 = px.line(full_df, x='year', y='area_km2', color='type', markers=True, title="Forecast (Polynomial Regression)")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📊 ARIMA Time Series Forecast (next 10 years)")
        try:
            model_arima = ARIMA(df_model['area_km2'], order=(1,1,1))
            model_fit = model_arima.fit()
            forecast = model_fit.forecast(steps=10)
            future_years_arima = np.arange(df_model['year'].iloc[-1]+1, df_model['year'].iloc[-1]+11)
            arima_df = pd.DataFrame({'year': future_years_arima, 'area_km2': forecast, 'type': 'ARIMA Forecast'})

            all_df = pd.concat([df_model[['year', 'area_km2', 'type']], arima_df])
            fig3 = px.line(all_df, x='year', y='area_km2', color='type', title="ARIMA Forecast")
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.warning("⚠️ ARIMA model failed.")

    elif page == "Alerts":
        st.title("🚨 Glacier Risk Alerts")
        latest_area = df['area_km2'].iloc[-1]
        threshold = 20.0
        if latest_area < threshold:
            st.error(f"🔴 Critical Alert! Area critically low ({latest_area:.2f} sq.km)")
        elif latest_area < threshold + 5:
            st.warning(f"🟡 Warning: Glacier nearing danger ({latest_area:.2f} sq.km)")
        else:
            st.success(f"🟢 Glacier stable. Current: {latest_area:.2f} sq.km")

    elif page == "Map Overview":
        st.title("🗺 Gangotri Glacier Map Overview")
        m = leafmap.Map(center=[30.96, 79.08], zoom=11)
        m.to_streamlit(height=600)

# ------------------- FOOTER -------------------
st.markdown("""
---
<p style="text-align: center; font-size: 14px;">
Made with ❤️ by Parkavi | Final Year Project 2025 | [GitHub](https://github.com/Parkavi-29)
</p>
""", unsafe_allow_html=True)
