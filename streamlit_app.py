import streamlit as st
import pandas as pd
import plotly.express as px
import leafmap.foliumap as leafmap
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Config
st.set_page_config(page_title="Glacier Melt Dashboard", layout="wide")

# Background Styling
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://img.freepik.com/free-photo/beautiful-scenery-summit-mount-everest-covered-with-snow-white-clouds_181624-21317.jpg?semt=ais_hybrid&w=740");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.88);
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("🧊 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# Load Data
csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Glacier_Area_Elevation_Trend_2001_2023.csv'
try:
    df = pd.read_csv(csv_url)
    st.success("✅ Data loaded from GitHub!")
except Exception as e:
    st.error("❌ Failed to load CSV data.")
    st.exception(e)
    df = None

# ------------------------- Pages ---------------------------
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
        st.title("📊 Glacier Area Forecast (2030 & 2050)")

        df_clean = df.dropna(subset=['year', 'area_km2'])
        X = df_clean['year'].values.reshape(-1, 1)
        y = df_clean['area_km2'].values.reshape(-1, 1)

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        # Forecast to 2030 and 2050
        future_years = np.array([2025, 2030, 2040, 2050]).reshape(-1, 1)
        future_poly = poly.transform(future_years)
        predictions = model.predict(future_poly).flatten()

        for yr, pred in zip(future_years.flatten(), predictions):
            st.metric(f"Predicted Glacier Area in {yr}", f"{max(0, pred):.2f} sq.km")

        # Combine data for plotting
        future_df = pd.DataFrame({
            'year': future_years.flatten(),
            'area_km2': np.maximum(predictions, 0),
            'type': 'Predicted'
        })

        df_clean['type'] = 'Observed'
        combined = pd.concat([df_clean[['year', 'area_km2', 'type']], future_df])

        fig_pred = px.line(combined, x='year', y='area_km2', color='type', markers=True,
                           title="Glacier Area Trend with Forecast (to 2050)",
                           labels={"year": "Year", "area_km2": "Area (sq.km)"})
        st.plotly_chart(fig_pred, use_container_width=True)

    elif page == "Alerts":
        st.title("🚨 Glacier Risk Alerts")
        threshold = 160.0
        current = df['area_km2'].iloc[-1]
        if current < threshold:
            st.error(f"🚨 ALERT: Glacier area dropped below {threshold} sq.km! Current: {current:.2f} sq.km")
        else:
            st.success("✅ Glacier area is currently safe.")

    elif page == "Map Overview":
        st.title("🗺️ Glacier Region Map Overview")
        m = leafmap.Map(center=[30.95, 79.05], zoom=10)
        m.to_streamlit(height=600)
