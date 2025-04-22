import streamlit as st
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(page_title="Glacier Melt Analysis", layout="wide")

# Title and Description
st.title("🧊 Glacier Melt Analysis")
st.markdown("""
Welcome to your final year project! This app visualizes glacier melt data over time, 
based on satellite analysis from Google Earth Engine.
""")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your Glacier Area CSV (from GEE)", type="csv")

# If file is uploaded
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Show data
    st.subheader("📋 Glacier Area Data (sq.km)")
    st.dataframe(df)

    # Plot glacier area over time
    st.subheader("📉 Glacier Area Over the Years")
    fig = px.line(df, x='year', y='area_km2', markers=True,
                  title="Glacier Retreat Trend",
                  labels={"area_km2": "Glacier Area (sq.km)", "year": "Year"})
    st.plotly_chart(fig, use_container_width=True)

    # Calculate loss
    initial = df['area_km2'].max()
    latest = df['area_km2'].min()
    loss = initial - latest

    st.metric("📉 Total Glacier Loss (2015–2023)", f"{loss:.2f} sq.km")

    # Download button
    st.download_button("📥 Download CSV", uploaded_file, file_name="Glacier_Area_Trend.csv")

else:
    st.info("👈 Upload a CSV to get started. You can use your export from Google Earth Engine.")


