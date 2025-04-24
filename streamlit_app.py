import streamlit as st
import pandas as pd
import plotly.express as px
import leafmap.foliumap as leafmap
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Set Streamlit config
st.set_page_config(page_title="Glacier Melt Dashboard", layout="wide")

# Background image
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.pexels.com/photos/1416900/pexels-photo-1416900.jpeg?cs=srgb&dl=pexels-rasikraj-1416900.jpg&fm=jpg");
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
st.sidebar.title("🗊 Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# Load data
csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Glacier_Area_Elevation_Trend_2001_2023.csv'
try:
    df = pd.read_csv(csv_url)
    st.success("✅ Data loaded from GitHub!")
except Exception as e:
    st.error("❌ Failed to load CSV data.")
    st.exception(e)
    df = None

# ------------------------ Pages ------------------------
if df is not None:
    if page == "Overview":
        st.title("📋 Glacier Melt Analysis Web App")
        st.markdown("This app visualizes glacier retreat and elevation trends using GEE data (2001–2023).")
        st.dataframe(df)

    elif page == "Chart View":
        st.title("📈 Glacier Trend Charts")
        fig_area = px.line(df, x='year', y='area_km2', markers=True,
                           title="Retreat Trend",
                           labels={"year": "Year", "area_km2": "Area (sq.km)"})
        st.plotly_chart(fig_area, use_container_width=True)

        if 'mean_elevation_m' in df.columns:
            fig_elev = px.line(df, x='year', y='mean_elevation_m', markers=True,
                               title="Elevation Change",
                               labels={"year": "Year", "mean_elevation_m": "Elevation (m)"})
            st.plotly_chart(fig_elev, use_container_width=True)

        st.metric("📉 Total Glacier Loss", f"{df['area_km2'].max() - df['area_km2'].min():.2f} sq.km")
        if 'mean_elevation_m' in df.columns:
            st.metric("📈 Elevation Change", f"{df['mean_elevation_m'].iloc[-1] - df['mean_elevation_m'].iloc[0]:.2f} m")

        st.download_button("📅 Download CSV", df.to_csv(index=False), file_name="Glacier_Area_Trend.csv")

    elif page == "Prediction":
        st.title("📊 Future Glacier Area Prediction")

        if 'year' in df.columns and 'area_km2' in df.columns:
            df_clean = df[df['area_km2'] > 0].dropna(subset=['year', 'area_km2'])
            X = df_clean['year'].values.reshape(-1, 1)
            y = df_clean['area_km2'].values.reshape(-1, 1)

            # Polynomial Regression
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)

            # Future years: every 5 years from 2025 to 2050
            future_years = np.arange(2025, 2051, 5).reshape(-1, 1)
            future_poly = poly.transform(future_years)
            predictions = model.predict(future_poly)

            # Show metrics
            for year, pred in zip(future_years.flatten(), predictions.flatten()):
                st.metric(f"📊 Predicted Glacier Area ({year})", f"{pred:.2f} sq.km")

            # Combine for plotting
            future_df = pd.DataFrame({
                'year': future_years.flatten(),
                'area_km2': predictions.flatten(),
                'type': 'Predicted'
            })

            df_clean['type'] = 'Observed'
            combined_df = pd.concat([df_clean[['year', 'area_km2', 'type']], future_df])

            # Plot
            fig_pred = px.line(combined_df, x='year', y='area_km2', color='type', markers=True,
                               title="Glacier Area Trend with Forecast (to 2050)",
                               labels={"year": "Year", "area_km2": "Area (sq.km)"})
            st.plotly_chart(fig_pred, use_container_width=True)

        else:
            st.warning("❗ Required columns 'year' and 'area_km2' not found in CSV.")

    elif page == "Alerts":
        st.title("🚨 Glacier Risk Alerts")
        critical_threshold = 160.0
        current_area = df['area_km2'].iloc[-1]
        if current_area < critical_threshold:
            st.error(f"🚨 ALERT: Glacier area dropped below {critical_threshold} sq.km! Current: {current_area:.2f} sq.km")
        else:
            st.success("✅ Glacier area is safe.")
        st.markdown("📨 Future: Email/SMS alerts integration.")

    elif page == "Map Overview":
        st.title("🗺️ Glacier Region Map Overview")
        st.markdown("Map centered around Gangotri glacier.")
        m = leafmap.Map(center=[30.95, 79.05], zoom=10)
        m.to_streamlit(height=600)
