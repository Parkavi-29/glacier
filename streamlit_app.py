import streamlit as st
import pandas as pd
import plotly.express as px
import leafmap.foliumap as leafmap

# Set Streamlit config
st.set_page_config(page_title="Glacier Melt Dashboard", layout="wide")

# ✅ Background image + layout style
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.peakpx.com%2Fen%2Fhd-wallpaper-desktop-fixmk&psig=AOvVaw3x8mCm_VP17_vadFKLAfpj&ust=1745556136721000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCMiS88Dt74wDFQAAAAAdAAAAABAR");
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

# Sidebar navigation
st.sidebar.title("🧊 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# ✅ Load updated CSV (2001–2023)
csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Glacier_Area_Elevation_Trend_2001_2023.csv'

try:
    df = pd.read_csv(csv_url)
    st.success("✅ Data loaded from GitHub!")
except Exception as e:
    st.error("❌ Failed to load CSV data.")
    st.exception(e)
    df = None

# -------------------------------
# 🧊 Pages
# -------------------------------
if df is not None:
    if page == "Overview":
        st.title("📋 Glacier Melt Analysis Web App")
        st.markdown("This dashboard visualizes glacier retreat and elevation trends (2001–2023) using Google Earth Engine (GEE).")
        st.dataframe(df)

    elif page == "Chart View":
        st.title("📈 Glacier Trend Charts")
        fig_area = px.line(df, x='year', y='area_km2', markers=True,
                           title="Glacier Retreat Over Time",
                           labels={"year": "Year", "area_km2": "Area (sq.km)"})
        st.plotly_chart(fig_area, use_container_width=True)

        if 'mean_elevation_m' in df.columns:
            fig_elev = px.line(df, x='year', y='mean_elevation_m', markers=True,
                               title="Mean Elevation of Glacier (m)",
                               labels={"year": "Year", "mean_elevation_m": "Elevation (m)"})
            st.plotly_chart(fig_elev, use_container_width=True)

        st.metric("📉 Total Glacier Loss", f"{df['area_km2'].max() - df['area_km2'].min():.2f} sq.km")
        if 'mean_elevation_m' in df.columns:
            st.metric("📈 Elevation Change", f"{df['mean_elevation_m'].iloc[-1] - df['mean_elevation_m'].iloc[0]:.2f} m")

        st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="Glacier_Area_Elevation_Trend_2001_2023.csv")

    elif page == "Prediction":
        st.title("📊 Future Glacier Area Prediction")
        st.info("🔄 Coming soon: ML-based forecasts for 2025 & 2030.")

    elif page == "Alerts":
        st.title("🚨 Glacier Risk Alerts")
        critical_threshold = 160.0
        current_area = df['area_km2'].iloc[-1]
        if current_area < critical_threshold:
            st.error(f"🚨 ALERT: Glacier area dropped below {critical_threshold} sq.km! Current: {current_area:.2f} sq.km")
        else:
            st.success("✅ Glacier area is safe.")
        st.markdown("📨 Future: Email/SMS alert system integration.")

    elif page == "Map Overview":
        st.title("🗺️ Glacier Region Map Overview")
        st.markdown("Interactive map centered on Gangotri Glacier.")
        m = leafmap.Map(center=[30.95, 79.05], zoom=10)
        m.to_streamlit(height=600)
