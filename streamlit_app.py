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
st.set_page_config(layout="wide")

# ------------------- CLOCK (PRO STYLE) -------------------
ist = pytz.timezone('Asia/Kolkata')
now = datetime.now(ist)

current_time = now.strftime('%I:%M %p')  # 04:26 PM
current_date = now.strftime('%A, %d %B %Y')  # Saturday, 27 April 2025

# ------------------- SIDEBAR NAV -------------------
st.sidebar.title("🧨 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# ------------------- STYLING -------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
<style>
[data-testid="stAppViewContainer"] {
    background-color: #F5F7FA;
    font-family: 'Poppins', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# ------------------- MAIN 3 COLUMN LAYOUT -------------------
col1, col2, col3 = st.columns([2, 1.2, 1.5])

# --------- COLUMN 1 (Main Pages) ---------
with col1:
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

# --------- COLUMN 2 (Gangotri Images) ---------
with col2:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcREy-RSzTOCMqJTuRwLLiwr7UpAAGv6Hpv6xQ&s", caption="Gangotri Glacier View 1", use_column_width=True)
    st.image("https://www.remotelands.com/travelogues/app/uploads/2018/06/DSC00976-2.jpg", caption="Gangotri Glacier View 2", use_column_width=True)
    st.image("https://akm-img-a-in.tosshub.com/aajtak/images/story/202209/gaumukh_gangotri_glacier_getty_1-sixteen_nine.jpg?size=948:533", caption="Gaumukh Source", use_column_width=True)

# --------- COLUMN 3 (Date/Time and Chatbot) ---------
with col3:
    # --------- CLOCK ---------
    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0;">
        <div style="font-size: 60px; font-weight: bold; 
                    background: linear-gradient(90deg, #1E90FF, #8A2BE2);
                    -webkit-background-clip: text;
                    color: transparent;
                    font-family: 'Poppins', sans-serif;">
            {current_time}
        </div>
        <div style="font-size: 24px; color: #666666; font-family: 'Poppins', sans-serif;">
            {current_date}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --------- CHATBOT ---------
    with st.expander("💬 Ask GlacierBot", expanded=True):
        user_q = st.text_input("Your question:", placeholder="e.g. What is NDSI?", key="chat")

        if user_q:
            q = user_q.lower()
            if "ndsi" in q:
                st.success("🧊 NDSI stands for Normalized Difference Snow Index, used to detect snow and ice in satellite images.")
            elif "gangotri" in q:
                st.success("🗻 The Gangotri Glacier is one of the largest glaciers in the Himalayas and source of the Ganges.")
            elif "retreat" in q:
                st.success("📉 Glacier retreat refers to the shrinking of glaciers due to melting over time.")
            elif "area" in q:
                st.success("🗺 Area is calculated by detecting glacier pixels using NDSI threshold > 0.4.")
            elif "arima" in q:
                st.success("📊 ARIMA is a time series forecasting model used for glacier area prediction.")
            elif "regression" in q:
                st.success("📉 Polynomial regression helps model glacier area trends over years.")
            else:
                st.info("🤖 I'm still learning. Try asking about glaciers, ARIMA, NDSI, etc.")
