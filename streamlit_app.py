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

# Set timezone to IST
ist = pytz.timezone('Asia/Kolkata')
current_time_ist = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')

# Real-time Clock in IST
st.markdown(f"🕒 *Current Date & Time (IST):* {current_time_ist}")

# Background + font adjustments
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://w0.peakpx.com/wallpaper/262/173/HD-wallpaper-samsung-background-blue-edge-gradient-gray-plain-purple-simple-sky-thumbnail.jpg");
    background-size: cover;
    background-attachment: fixed;
    color: #2e2e2e;
}
.main {
    background-color: rgba(255, 255, 255, 0.88);
    padding: 2rem;
    border-radius: 10px;
}
h1, h2, h3 {
    color: #0b3954 !important;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.75);
}
</style>
""", unsafe_allow_html=True)


# Sidebar
st.sidebar.title("🧊 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# Load Data
csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Glacier_Area_Elevation_Trend_2001_2023.csv'
try:
    df = pd.read_csv(csv_url)
    df = df.dropna(subset=['year', 'area_km2'])  # Clean
    st.success("✅ Data loaded from GitHub!")
except Exception as e:
    st.error("❌ Failed to load CSV data.")
    st.exception(e)
    df = None

# Content
if df is not None:
    if page == "Overview":
        st.title("📋 Glacier Melt Analysis Web App")
        st.markdown("This app visualizes glacier retreat and elevation trends using GEE data (2001–2023).")
        st.dataframe(df)

    elif page == "Chart View":
        st.title("📈 Glacier Trend Charts")
        fig_area = px.line(df, x='year', y='area_km2', markers=True, title="Retreat Trend")
        st.plotly_chart(fig_area, use_container_width=True)

        if 'mean_elevation_m' in df.columns:
            fig_elev = px.line(df, x='year', y='mean_elevation_m', markers=True, title="Elevation Trend")
            st.plotly_chart(fig_elev, use_container_width=True)

        st.metric("📉 Total Glacier Loss", f"{df['area_km2'].max() - df['area_km2'].min():.2f} sq.km")

    elif page == "Prediction":
        st.title("🔮 Future Glacier Area Prediction")

        df_model = df.copy()
        X = df_model['year'].values.reshape(-1, 1)
        y = df_model['area_km2'].values.reshape(-1, 1)

        st.subheader("📉 Polynomial Regression Forecast (to 2050)")
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
        fig = px.line(full_df, x='year', y='area_km2', color='type', markers=True, title="Polynomial Regression Forecast")
        st.plotly_chart(fig, use_container_width=True)

        # Display metrics
        for y, p in zip(future_years.flatten(), pred_poly.flatten()):
            st.metric(f"📈 Predicted Area ({y})", f"{p:.2f} sq.km")

        # Time Series Forecast (ARIMA)
        st.subheader("📊 ARIMA Time Series Forecast (next 10 years)")
        try:
            model_arima = ARIMA(df_model['area_km2'], order=(1, 1, 1))
            model_fit = model_arima.fit()
            forecast = model_fit.forecast(steps=10)
            future_years_arima = np.arange(df_model['year'].iloc[-1]+1, df_model['year'].iloc[-1]+11)
            arima_df = pd.DataFrame({'year': future_years_arima, 'area_km2': forecast, 'type': 'ARIMA Forecast'})
            all_df = pd.concat([df_model[['year', 'area_km2', 'type']], arima_df])
            fig_arima = px.line(all_df, x='year', y='area_km2', color='type', title="ARIMA Forecast")
            st.plotly_chart(fig_arima, use_container_width=True)
        except Exception as e:
            st.warning("⚠️ ARIMA failed to fit. Try different parameters.")

    elif page == "Alerts":
        st.title("🚨 Glacier Risk Alerts")
        latest_area = df['area_km2'].iloc[-1]
        threshold = 160.0
        if latest_area < threshold:
            st.error(f"⚠️ Critical Alert! Glacier area has fallen below {threshold} sq.km. Current: {latest_area:.2f}")
        elif latest_area < threshold + 20:
            st.warning(f"🟡 Moderate Risk: Glacier area nearing critical threshold. Current: {latest_area:.2f}")
        else:
            st.success(f"🟢 Safe: Current glacier area is {latest_area:.2f} sq.km")

    elif page == "Map Overview":
        st.title("🗺️ Gangotri Glacier Map")
        m = leafmap.Map(center=[30.95, 79.05], zoom=10)
        m.to_streamlit(height=600)
