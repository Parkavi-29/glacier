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

# ------------------- Setup -------------------
ist = pytz.timezone('Asia/Kolkata')
current_time_ist = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S').upper()
st.set_page_config(layout="wide")

# ------------------- Styling -------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Catamaran:wght@400;600;700&display=swap" rel="stylesheet">
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.pinimg.com/736x/f4/53/7f/f4537f4d86850471d5642c7beea07bbd.jpg");
    background-size: cover;
    background-attachment: fixed;
    font-family: 'Catamaran', sans-serif;
}
.main {
    background-color: rgba(255, 255, 255, 0.88);
    padding: 2rem;
    border-radius: 10px;
    font-family: 'Catamaran', sans-serif;
}
h1, h2, h3 {
    color: #0b3954 !important;
    font-family: 'Catamaran', sans-serif;
    font-weight: 700;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.75);
    font-family: 'Catamaran', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ------------------- Sidebar Navigation -------------------
st.sidebar.title("🧨 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview", "Chatbot"])

# ------------------- Data -------------------
csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Gangotri_Glacier_Area_NDSI_2001_2023.csv'
try:
    df = pd.read_csv(csv_url)
    df = df.dropna(subset=['year', 'area_km2'])
    st.success("✅ Data loaded from GitHub!")
except Exception as e:
    st.error("❌ Failed to load CSV data.")
    st.exception(e)
    df = None

# ------------------- Pages -------------------
if df is not None:
    if page == "Overview":
        st.title("📋 Glacier Melt Analysis (Gangotri)")
        st.markdown("Analyzing Gangotri Glacier retreat from Landsat Data (NDSI-based, 2001–2023)")
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

        pred_df = pd.DataFrame({'year': future_years.flatten(), 'area_km2': pred_poly.flatten(), 'type': 'Predicted'})
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
            st.warning("⚠️ ARIMA forecast failed. Try changing the model parameters.")

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

    elif page == "Chatbot":
        st.title("🤖 Glacier Assistant Chatbot")
        st.markdown("Ask anything about glaciers. Try: *What is a glacier?*, *What is NDSI?*, etc.")

        user_input = st.text_input("Ask your question:")
        if user_input:
            question = user_input.lower()

            # Hardcoded answers
            responses = {
                "what is a glacier": "A glacier is a large mass of ice that forms over many years and flows slowly over land.",
                "what is glacier retreat": "Glacier retreat refers to the shrinking of glacier size due to melting.",
                "what is gangotri glacier": "Gangotri Glacier is one of the largest Himalayan glaciers and source of the Ganges.",
                "what is ndsi": "NDSI stands for Normalized Difference Snow Index. It's used to identify snow and ice using satellite data.",
                "what is landsat": "Landsat is a series of Earth-observing satellites jointly managed by NASA and USGS.",
                "how is glacier area calculated": "We use satellite images (e.g. Landsat) and indices like NDSI to detect glacier area.",
                "why is glacier melting": "Due to rising global temperatures and climate change.",
                "what is polynomial regression": "A type of regression used to model curved relationships like glacier melting trends.",
                "what is arima": "ARIMA is a time series forecasting technique used for predicting future values.",
                "how can we save glaciers": "By reducing carbon emissions, using renewable energy, and conserving water and forests."
            }

            matched = False
            for q, ans in responses.items():
                if q in question:
                    st.success(f"💬 {ans}")
                    matched = True
                    break

            if not matched:
                st.info("🤔 I'm still learning. Try simpler questions like 'What is a glacier?'")

# Footer IST time
st.markdown(f"""
<hr>
<p style='text-align:center;font-size:14px;'>🕒 Current IST Time: <strong>{current_time_ist}</strong></p>
""", unsafe_allow_html=True)
