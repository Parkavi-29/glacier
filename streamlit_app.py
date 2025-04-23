import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Glacier Melt Analysis", layout="wide")
st.title("🧊 Glacier Melt Analysis Web App")
st.markdown("This app visualizes glacier retreat and elevation trends using GEE data (2015–2023).")

csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Glacier_Area_Trend.csv'

try:
    df = pd.read_csv(csv_url)
    st.success("✅ Data loaded from GitHub!")

    st.subheader("📋 Glacier Data Table")
    st.dataframe(df)

    # Area Trend
    st.subheader("📉 Glacier Area Over the Years")
    fig_area = px.line(df, x='year', y='area_km2', markers=True,
                       title="Glacier Retreat Trend",
                       labels={"year": "Year", "area_km2": "Area (sq.km)"})
    st.plotly_chart(fig_area, use_container_width=True)

    # Elevation Trend
    if 'mean_elevation_m' in df.columns:
        st.subheader("⛰️ Mean Elevation of Glacier (m)")
        fig_elev = px.line(df, x='year', y='mean_elevation_m', markers=True,
                           title="Mean Glacier Elevation Trend",
                           labels={"year": "Year", "mean_elevation_m": "Elevation (m)"})
        st.plotly_chart(fig_elev, use_container_width=True)

    # Summary Metrics
    st.metric("📉 Total Glacier Loss", f"{df['area_km2'].max() - df['area_km2'].min():.2f} sq.km")
    if 'mean_elevation_m' in df.columns:
        st.metric("📈 Elevation Increase", f"{df['mean_elevation_m'].iloc[-1] - df['mean_elevation_m'].iloc[0]:.2f} m")

    # Download CSV
    st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="Glacier_Area_Trend.csv", mime="text/csv")

except Exception as e:
    st.error("⚠️ Could not load or parse CSV.")
    st.exception(e)
