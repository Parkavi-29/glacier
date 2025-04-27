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

current_time = now.strftime('%I:%M %p')  # Example: 01:59 PM
current_date = now.strftime('%A, %d %B %Y')  # Example: Sunday, 27 April 2025

st.set_page_config(page_title="Gangotri Glacier Dashboard", layout="wide")

# ------------------- STYLING -------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://upload.wikimedia.org/wikipedia/commons/3/39/Gangotri_Glacier.jpg");
    background-size: cover;
    background-attachment: fixed;
    font-family: 'Poppins', sans-serif;
}
.main {
    background-color: rgba(255, 255, 255, 0.9);
    padding: 2rem;
    border-radius: 10px;
}
h1, h2, h3 {
    color: #0b3954 !important;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.8);
}
</style>
""", unsafe_allow_html=True)

# ------------------- HEADER -------------------
st.markdown(f"""
<div style="text-align: center; padding: 10px; font-family: 'Poppins', sans-serif;">
    <div style="font-size: 65px; font-weight: bold; color: #003566;">🧊 {current_time}</div>
    <div style="font-size: 28px; font-weight: 500; color: #003566;">{current_date}</div>
</div>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR NAV -------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# ------------------- CHATBOT -------------------
with st.sidebar.expander("💬 GlacierBot - Ask Me!"):
    st.markdown("""Friendly assistant! Try asking about NDSI, Gangotri Glacier, Retreat, ARIMA etc.""")
    user_q = st.text_input("Your question:", placeholder="Type here...")
    if user_q:
        q = user_q.lower()
        if "ndsi" in q:
            st.success("🧊 NDSI stands for Normalized Difference Snow Index, used to detect snow and ice.")
        elif "gangotri" in q:
            st.success("🗻 Gangotri Glacier is one of the largest in the Himalayas, source of the Ganges River.")
        elif "retreat" in q:
            st.success("📉 Glacier retreat means the glacier is shrinking over time.")
        elif "climate" in q:
            st.success("🌡 Climate change is the main cause behind glacier melting.")
        else:
            st.info("🤖 I'm learning! Ask about glaciers, NDSI, temperature effects etc.")

# ------------------- DATA LOADING -------------------
csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Gangotri_Glacier_Area_NDSI_2001_2023.csv'
try:
    df = pd.read_csv(csv_url)
    df = df.dropna(subset=['year', 'area_km2'])
    st.success("✅ Data loaded successfully!")
except Exception as e:
    st.error("❌ Error loading data.")
    st.exception(e)
    df = None

# ------------------- PAGES -------------------
if df is not None:
    if page == "Overview":
        st.title("📋 Glacier Overview")
        st.markdown("Analyzing Gangotri Glacier (2001-2023) using Landsat data and Machine Learning models.")
        st.dataframe(df, use_container_width=True)

    elif page == "Chart View":
        st.title("📈 Glacier Area Trends")
        fig = px.line(df, x='year', y='area_km2', markers=True, title="Observed Glacier Area 2001-2023", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Prediction":
        st.title("🔮 Predict Future Glacier Area")
        df_model = df.copy()
        X = df_model['year'].values.reshape(-1, 1)
        y = df_model['area_km2'].values.reshape(-1, 1)

        st.subheader("📉 Polynomial Regression")
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)

        future_years = np.arange(2025, 2051, 5).reshape(-1, 1)
        future_preds = model.predict(poly.transform(future_years))

        pred_df = pd.DataFrame({
            'year': future_years.flatten(),
            'area_km2': future_preds.flatten(),
            'type': 'Predicted'
        })

        df_model['type'] = 'Observed'
        combined_df = pd.concat([df_model[['year', 'area_km2', 'type']], pred_df])

        fig_pred = px.line(combined_df, x='year', y='area_km2', color='type', markers=True, template="plotly_white")
        st.plotly_chart(fig_pred, use_container_width=True)

    elif page == "Alerts":
        st.title("🚨 Risk Monitoring")
        latest_area = df['area_km2'].iloc[-1]
        threshold = 20.0
        if latest_area < threshold:
            st.error(f"🔴 Danger: Glacier critically low! ({latest_area:.2f} sq.km)")
        elif latest_area < threshold + 5:
            st.warning(f"🟡 Warning: Glacier nearing danger zone. ({latest_area:.2f} sq.km)")
        else:
            st.success(f"🟢 Glacier stable. Current area: {latest_area:.2f} sq.km")

    elif page == "Map Overview":
        st.title("🗺 Gangotri Glacier Map")
        m = leafmap.Map(center=[30.96, 79.08], zoom=11)
        m.to_streamlit(height=600)

# ------------------- END -------------------
