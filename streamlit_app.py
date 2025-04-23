import streamlit as st
import pandas as pd
import plotly.express as px
import leafmap.foliumap as leafmap

# Setup page
st.set_page_config(page_title="Glacier Melt Dashboard", layout="wide")

# Background image style
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://c4.wallpaperflare.com/wallpaper/384/818/513/himalayas-mountains-landscape-nature-wallpaper-preview.jpg");
        background-size: cover;
        background-attachment: fixed;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.92);
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
st.sidebar.title("🧊 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# Load data
csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Glacier_Area_Trend.csv'
try:
    df = pd.read_csv(csv_url)
except Exception as e:
    st.error("❌ Failed to load CSV data.")
    st.exception(e)
    df = None

# ------------------------
# Pages Logic
# ------------------------
if df is not None:
    if page == "Overview":
        st.title("📋 Glacier Melt Analysis Web App")
        st.markdown("This app visualizes glacier retreat and elevation trends using GEE data (2001–2023).")
        st.success("✅ Data loaded from GitHub!")
        st.dataframe(df)

    elif page == "Chart View":
        st.title("📈 Glacier Trend Charts")
        st.subheader("📉 Glacier Area Over the Years")
        fig_area = px.line(df, x='year', y='area_km2', markers=True,
                           title="Retreat Trend",
                           labels={"year": "Year", "area_km2": "Area (sq.km)"})
        st.plotly_chart(fig_area, use_container_width=True)

        if 'mean_elevation_m' in df.columns:
            st.subheader("⛰️ Mean Elevation Trend")
            fig_elev = px.line(df, x='year', y='mean_elevation_m', markers=True,
                               title="Elevation Change",
                               labels={"year": "Year", "mean_elevation_m": "Elevation (m)"})
            st.plotly_chart(fig_elev, use_container_width=True)

        st.metric("📉 Total Glacier Loss", f"{df['area_km2'].max() - df['area_km2'].min():.2f} sq.km")
        if 'mean_elevation_m' in df.columns:
            st.metric("📈 Elevation Change", f"{df['mean_elevation_m'].iloc[-1] - df['mean_elevation_m'].iloc[0]:.2f} m")

        st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="Glacier_Area_Trend.csv", mime="text/csv")

    elif page == "Prediction":
        st.title("📊 Future Glacier Area Prediction")
        st.info("🔄 Prediction module coming soon: will include ML-based regression.")
        st.markdown("📅 Forecasts for 2025 and 2030 will be generated using trends and models.")

    elif page == "Alerts":
        st.title("🚨 Glacier Risk Alerts")
        critical_threshold = 160.0
        current_area = df['area_km2'].iloc[-1]
        if current_area < critical_threshold:
            st.error(f"🚨 ALERT: Glacier area has dropped below {critical_threshold} sq.km! Current: {current_area:.2f} sq.km")
        else:
            st.success("✅ Glacier area is currently above critical threshold.")
        st.markdown("📨 In future, this will include email/SMS notifications for high-risk changes.")

    elif page == "Map Overview":
        st.title("🗺️ Glacier Region Map Overview")
        st.markdown("Visualizing glacier mask region with base map context.")
        m = leafmap.Map(center=[30.95, 79.05], zoom=10)
        m.to_streamlit(height=600)
