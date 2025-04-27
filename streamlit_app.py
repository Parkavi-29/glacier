# 🚀 FINAL PRO VERSION - 3 COLUMNS LAYOUT

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
now = datetime.now(ist)
current_time = now.strftime('%-I:%M')
current_date = now.strftime('%a, %d %B')

st.set_page_config(layout="wide")

# ------------------- CLOCK -------------------
st.markdown(f"""
<div style="text-align: center; color: white; padding: 10px; font-family: 'Catamaran', sans-serif;">
    <div style="font-size: 70px; font-weight: bold;">{current_time}</div>
    <div style="font-size: 30px;">{current_date}</div>
</div>
""", unsafe_allow_html=True)

# ------------------- STYLING -------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Catamaran:wght@400;600;700&display=swap" rel="stylesheet">
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f0f2f6;
    background-attachment: fixed;
    font-family: 'Catamaran', sans-serif;
}
.main {
    background-color: rgba(255, 255, 255, 0.9);
    padding: 2rem;
    border-radius: 10px;
}
h1, h2, h3 {
    color: #0b3954 !important;
    font-family: 'Catamaran', sans-serif;
    font-weight: 700;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.85);
}
</style>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR NAV -------------------
st.sidebar.title("🌨 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# ------------------- LAYOUT: 3 COLUMNS -------------------
col_main, col_gallery, col_chatbot = st.columns([3, 1.5, 1.5])

# ------------------- MAIN CONTENT -------------------
with col_main:
    csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Gangotri_Glacier_Area_NDSI_2001_2023.csv'
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

# ------------------- GALLERY -------------------
with col_gallery:
    st.header("🖼 Gangotri Glacier Views")
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcREy-RSzTOCMqJTuRwLLiwr7UpAAGv6Hpv6xQ&s", caption="Gangotri Glacier View 1")
    st.image("https://www.remotelands.com/travelogues/app/uploads/2018/06/DSC00976-2.jpg", caption="Gangotri Glacier View 2")
    st.image("https://akm-img-a-in.tosshub.com/aajtak/images/story/202209/gaumukh_gangotri_glacier_getty_1-sixteen_nine.jpg?size=948:533", caption="Gangotri Glacier View 3")

# ------------------- CHATBOT -------------------
with col_chatbot:
    st.header("💬 Ask GlacierBot")
    user_q = st.text_area("Type your glacier question:", height=150, placeholder="e.g. What causes glacier melting?")
    if user_q:
        q = user_q.lower()
        if "ndsi" in q:
            st.success("🧊 NDSI is used to detect snow and ice based on green and SWIR bands.")
        elif "gangotri" in q:
            st.success("🗻 Gangotri Glacier is a primary source of the Ganges river.")
        elif "melting" in q or "retreat" in q:
            st.success("📉 Glacier melting is mainly due to rising temperatures and climate change.")
        elif "landsat" in q:
            st.success("🛰 Landsat 5, 7, and 8 images were used to analyze glacier retreat from 2001 to 2023.")
        else:
            st.info("🤖 Sorry, I know about NDSI, Gangotri, melting, Landsat. More topics soon!")

# 🚀 DONE! FINAL PRO 3-COLUMN VERSION!
