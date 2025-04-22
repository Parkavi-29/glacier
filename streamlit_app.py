import streamlit as st
import pandas as pd
import plotly.express as px

# Set page layout
st.set_page_config(page_title="Glacier Melt Analysis", layout="wide")

st.title("🧊 Glacier Melt Analysis Web App")
st.markdown("This app visualizes glacier retreat over time using data from Google Earth Engine.")

# Load CSV directly from GitHub
csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Glacier_Area_Trend.csv'

try:
    df = pd.read_csv(csv_url)
    st.success("✅ Data loaded from GitHub!")

    # Show raw data
    st.subheader("📋 Glacier Area Data (sq.km)")
    st.dataframe(df)

    # Line chart
    st.subheader("📉 Glacier Area Over the Years")
    fig = px.line(df, x='year', y='area_km2', markers=True,
                  title="Glacier Retreat Trend",
                  labels={"year": "Year", "area_km2": "Area (sq.km)"})
    st.plotly_chart(fig, use_container_width=True)

    # Area loss summary
    initial = df['area_km2'].max()
    latest = df['area_km2'].min()
    loss = initial - latest
    st.metric("📉 Total Glacier Loss (2015–2023)", f"{loss:.2f} sq.km")

    # Download button
    st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="Glacier_Area_Trend.csv", mime="text/csv")

except Exception as e:
    st.error("⚠️ Could not load CSV. Please make sure the file exists in your GitHub repo.")
    st.exception(e)
