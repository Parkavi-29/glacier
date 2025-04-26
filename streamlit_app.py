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
import openai

# ⏰ Real-time IST clock
ist = pytz.timezone('Asia/Kolkata')
current_time_ist = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S').upper()
st.markdown(f"<div style='font-size:24px;font-weight:bold;'>🕒 Current Date & Time (IST): {current_time_ist}</div>", unsafe_allow_html=True)

# 🎨 Background styling
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
h1, h2, h3 { color: #0b3954 !important; }
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.75);
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🧊 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview", "Chatbot"])

# 📥 Load Glacier CSV from GitHub
csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Gangotri_Glacier_Area_NDSI_2001_2023.csv'
try:
    df = pd.read_csv(csv_url)
    df = df.dropna(subset=['year', 'area_km2'])
    st.success("✅ Data loaded from GitHub!")
except Exception as e:
    st.error("❌ Failed to load CSV data.")
    st.exception(e)
    df = None

# ---------------------- Pages ----------------------
if df is not None:
    if page == "Overview":
        st.title("📋 Glacier Melt Analysis Web App")
        st.markdown("This app visualizes Gangotri glacier retreat (2001–2023) using GEE with NDSI.")
        st.dataframe(df)

    elif page == "Chart View":
        st.title("📈 Glacier Trend Charts")
        fig_area = px.line(df, x='year', y='area_km2', markers=True, title="Glacier Area Retreat")
        st.plotly_chart(fig_area, use_container_width=True)
        st.metric("📉 Total Glacier Loss", f"{df['area_km2'].max() - df['area_km2'].min():.2f} sq.km")

    elif page == "Prediction":
        st.title("🔮 Future Glacier Area Prediction")
        df_model = df.copy()
        X = df_model['year'].values.reshape(-1, 1)
        y = df_model['area_km2'].values.reshape(-1, 1)

        # Polynomial regression
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

        for year, val in zip(future_years.flatten(), pred_poly.flatten()):
            st.metric(f"📈 Predicted Area ({year})", f"{val:.2f} sq.km")

        # ARIMA forecast
        st.subheader("📊 ARIMA Time Series Forecast (next 10 years)")
        try:
            model_arima = ARIMA(df_model['area_km2'], order=(1, 1, 1))
            model_fit = model_arima.fit()
            forecast = model_fit.forecast(steps=10)
            future_arima = np.arange(df_model['year'].iloc[-1] + 1, df_model['year'].iloc[-1] + 11)
            arima_df = pd.DataFrame({'year': future_arima, 'area_km2': forecast, 'type': 'ARIMA Forecast'})

            all_df = pd.concat([df_model[['year', 'area_km2', 'type']], arima_df])
            fig_arima = px.line(all_df, x='year', y='area_km2', color='type',
                                title="ARIMA Forecast - Glacier Area")
            st.plotly_chart(fig_arima, use_container_width=True)
        except:
            st.warning("⚠️ ARIMA model failed. Try different parameters.")

    elif page == "Alerts":
        st.title("🚨 Glacier Risk Alerts")
        latest = df['area_km2'].iloc[-1]
        threshold = 20.0
        if latest < threshold:
            st.error(f"🔴 ALERT: Glacier area below {threshold} sq.km. Current: {latest:.2f}")
        elif latest < threshold + 5:
            st.warning(f"🟡 Warning: Area nearing threshold. Current: {latest:.2f}")
        else:
            st.success(f"🟢 Safe: Current glacier area = {latest:.2f} sq.km")

    elif page == "Map Overview":
        st.title("🗺️ Gangotri Glacier Map")
        m = leafmap.Map(center=[30.95, 79.07], zoom=11)
        m.to_streamlit(height=600)

    elif page == "Chatbot":
        st.title("💬 Ask GlacierBot (GPT)")
        st.markdown("A helpful assistant to explain glacier melt, GEE, predictions, and more.")

        openai.api_key = st.secrets["openai"]["api_key"]
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).markdown(msg["content"])

        prompt = st.chat_input("Ask about glacier trends, data, or GEE...")

        if prompt:
            st.chat_message("user").markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state.chat_history,
                )
                reply = response.choices[0].message["content"]
                st.chat_message("assistant").markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error("⚠️ GPT failed to respond.")
                st.exception(e)
