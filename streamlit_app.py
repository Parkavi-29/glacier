import streamlit as st
import pandas as pd
import plotly.express as px
import leafmap.foliumap as leafmap
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import datetime

# --- Page configuration ---
st.set_page_config(page_title="Glacier Melt Dashboard", layout="wide")

# --- Custom styles with background and fonts ---
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://w0.peakpx.com/wallpaper/262/173/HD-wallpaper-samsung-background-blue-edge-gradient-gray-plain-purple-simple-sky-thumbnail.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #2e2e2e;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.88);
        padding: 2rem;
        border-radius: 10px;
        color: #2e2e2e;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #0b3954 !important;
    }
    .stMetric {
        color: #003049 !important;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.75);
        color: #003049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar navigation ---
st.sidebar.title("🏔️ Glacier Dashboard")
st.sidebar.markdown("*Current Time:* " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# --- Load data from GitHub ---
csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Glacier_Area_Elevation_Trend_2001_2023.csv'
try:
    df = pd.read_csv(csv_url)
    df = df.dropna(subset=['year', 'area_km2'])
    st.success("✅ Data loaded from GitHub!")
except Exception as e:
    st.error("❌ Failed to load CSV data.")
    st.exception(e)
    df = None

if df is not None:
    if page == "Overview":
        st.title("📋 Glacier Melt Analysis")
        st.markdown("This app visualizes glacier retreat and elevation trends using GEE data (2001–2023).")
        st.dataframe(df)

    elif page == "Chart View":
        st.title("📈 Glacier Trend Charts")
        fig_area = px.line(df, x='year', y='area_km2', markers=True, color_discrete_sequence=['#00b4d8'])
        st.plotly_chart(fig_area, use_container_width=True)

        if 'mean_elevation_m' in df.columns:
            fig_elev = px.line(df, x='year', y='mean_elevation_m', markers=True, color_discrete_sequence=['#0077b6'])
            st.plotly_chart(fig_elev, use_container_width=True)

        st.metric("📉 Total Glacier Loss", f"{df['area_km2'].max() - df['area_km2'].min():.2f} sq.km")
        st.metric("📈 Elevation Change", f"{df['mean_elevation_m'].iloc[-1] - df['mean_elevation_m'].iloc[0]:.2f} m")

    elif page == "Prediction":
        st.title("🔮 Future Glacier Area Prediction")
        model = LinearRegression()
        X = df['year'].values.reshape(-1, 1)
        y = df['area_km2'].values.reshape(-1, 1)
        log_y = np.log(y.clip(min=1))
        model.fit(X, log_y)

        future_years = np.arange(2025, 2051, 5).reshape(-1, 1)
        log_preds = model.predict(future_years)
        predictions = np.exp(log_preds).clip(min=0)

        for year, pred in zip(future_years.flatten(), predictions.flatten()):
            st.metric(f"📈 Predicted Area ({year})", f"{pred:.2f} sq.km")

        # ARIMA Time Series Prediction
        st.subheader("📊 ARIMA Time Series Forecast")
        try:
            model_arima = ARIMA(df['area_km2'], order=(1, 1, 1))
            fit = model_arima.fit()
            forecast = fit.forecast(steps=6)
            forecast_years = list(range(2025, 2051, 5))
            arima_df = pd.DataFrame({'year': forecast_years, 'area_km2': forecast, 'type': 'ARIMA'})

            regression_df = pd.DataFrame({
                'year': future_years.flatten(),
                'area_km2': predictions.flatten(),
                'type': 'Regression'
            })

            combined_df = pd.concat([df[['year', 'area_km2']].assign(type='Observed'), regression_df, arima_df])
            fig = px.line(combined_df, x='year', y='area_km2', color='type', markers=True,
                          title="Glacier Area Forecast (Regression & ARIMA)")
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("ARIMA model could not be computed due to data limitations.")
    elif page == "Alerts":
        st.title("🚨 Glacier Risk Alerts")
        current_area = df['area_km2'].iloc[-1]
        critical_threshold = 160
        if current_area < critical_threshold:
            st.error(f"❗ ALERT: Glacier area dropped below {critical_threshold} sq.km (Current: {current_area:.2f})")
        else:
            st.success("✅ Glacier area is within safe limits.")

        st.metric("📆 Last Updated", df['year'].iloc[-1])
        st.metric("📏 Current Area", f"{current_area:.2f} sq.km")
        rate = df['area_km2'].diff().mean()
        risk = "High" if rate < -5 else ("Moderate" if rate < -2 else "Low")
        st.info(f"📉 Average Rate of Retreat: {rate:.2f} sq.km/year | Danger Level: *{risk}*")

    elif page == "Map Overview":
        st.title("🗺️ Gangotri Glacier Map")
        m = leafmap.Map(center=[30.95, 79.05], zoom=10)
        m.to_streamlit(height=600)

# --- Chatbot ---
with st.sidebar.expander("💬 Chat with GlacierBot"):
    user_msg = st.text_input("You:", placeholder="Ask about glacier retreat, area, or forecast")
    if user_msg:
        if "retreat" in user_msg.lower():
            st.write("🤖: Glacier retreat means the glacier is shrinking due to melting.")
        elif "elevation" in user_msg.lower():
            st.write("🤖: Mean elevation is the average altitude of the glacier region, influenced by climate.")
        elif "2030" in user_msg or "2050" in user_msg:
            st.write("🤖: Forecast models predict further decline in glacier area by 2030 and 2050.")
        else:
            st.write("🤖: I'm here to help! Ask me about glacier trends, area loss, or future predictions.")
