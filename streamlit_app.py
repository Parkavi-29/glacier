import streamlit as st
import pandas as pd
import plotly.express as px
import leafmap.foliumap as leafmap
import numpy as np
from sklearn.linear_model import LinearRegression

# Set page configuration
st.set_page_config(page_title="Glacier Melt Dashboard", layout="wide")

# Background and font style adjustments
st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("https://w0.peakpx.com/wallpaper/262/173/HD-wallpaper-samsung-background-blue-edge-gradient-gray-plain-purple-simple-sky-thumbnail.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #2e2e2e;
    }}
    .main {{
        background-color: rgba(255, 255, 255, 0.88);
        padding: 2rem;
        border-radius: 10px;
        color: #2e2e2e;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: #0b3954 !important;
    }}
    .stMetric {{
        color: #003049 !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: rgba(255, 255, 255, 0.75);
        color: #003049;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("\ud83c\udfca Glacier Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Chart View", "Prediction", "Alerts", "Map Overview"])

# Load data from GitHub
csv_url = 'https://raw.githubusercontent.com/Parkavi-29/glacier/main/Glacier_Area_Elevation_Trend_2001_2023.csv'
try:
    df = pd.read_csv(csv_url)
    st.success("\u2705 Data loaded from GitHub!")
except Exception as e:
    st.error("\u274c Failed to load CSV data.")
    st.exception(e)
    df = None

if df is not None:
    if page == "Overview":
        st.title("\ud83d\udccb Glacier Melt Analysis Web App")
        st.markdown("This app visualizes glacier retreat and elevation trends using GEE data (2001–2023).")
        st.dataframe(df)

    elif page == "Chart View":
        st.title("\ud83d\udcc8 Glacier Trend Charts")
        fig_area = px.line(df, x='year', y='area_km2', markers=True,
                           title="Retreat Trend",
                           labels={"year": "Year", "area_km2": "Area (sq.km)"})
        st.plotly_chart(fig_area, use_container_width=True)

        if 'mean_elevation_m' in df.columns:
            fig_elev = px.line(df, x='year', y='mean_elevation_m', markers=True,
                               title="Elevation Change",
                               labels={"year": "Year", "mean_elevation_m": "Elevation (m)"})
            st.plotly_chart(fig_elev, use_container_width=True)

        st.metric("\ud83d\udcc9 Total Glacier Loss", f"{df['area_km2'].max() - df['area_km2'].min():.2f} sq.km")
        if 'mean_elevation_m' in df.columns:
            st.metric("\ud83d\udcc8 Elevation Change", f"{df['mean_elevation_m'].iloc[-1] - df['mean_elevation_m'].iloc[0]:.2f} m")

        st.download_button("\ud83d\udcc5 Download CSV", df.to_csv(index=False), file_name="Glacier_Area_Trend.csv")

    elif page == "Prediction":
        st.title("\ud83d\udcca Future Glacier Area Prediction")

        if 'year' in df.columns and 'area_km2' in df.columns:
            df_clean = df.dropna(subset=['year', 'area_km2'])
            X = df_clean['year'].values.reshape(-1, 1)
            y = df_clean['area_km2'].values.reshape(-1, 1)

            log_y = np.log(y.clip(min=1))
            model = LinearRegression()
            model.fit(X, log_y)

            future_years = np.arange(2025, 2051, 5).reshape(-1, 1)
            log_pred = model.predict(future_years)
            predictions = np.exp(log_pred).clip(min=0)

            for year, pred in zip(future_years.flatten(), predictions.flatten()):
                st.metric(f"\ud83d\udcc8 Predicted Glacier Area ({year})", f"{pred:.2f} sq.km")

            future_df = pd.DataFrame({
                'year': future_years.flatten(),
                'area_km2': predictions.flatten(),
                'type': 'Predicted'
            })

            df_clean['type'] = 'Observed'
            combined_df = pd.concat([df_clean[['year', 'area_km2', 'type']], future_df])

            fig_pred = px.line(combined_df, x='year', y='area_km2', color='type', markers=True,
                               title="Glacier Area Trend with Forecast (to 2050)",
                               labels={"year": "Year", "area_km2": "Area (sq.km)"})
            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.warning("\u2757 Required columns 'year' and 'area_km2' not found in CSV.")

    elif page == "Alerts":
        st.title("\ud83d\udea8 Glacier Risk Alerts")
        critical_threshold = 100.0
        moderate_threshold = 150.0
        current_area = df['area_km2'].iloc[-1]
        change_rate = df['area_km2'].iloc[-1] - df['area_km2'].iloc[-2]

        st.info(f"\ud83d\udd52 Last Updated Year: {int(df['year'].iloc[-1])}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("\ud83d\udccd Current Glacier Area", f"{current_area:.2f} sq.km")
        with col2:
            st.metric("\ud83d\udcc9 Yearly Area Change", f"{change_rate:+.2f} sq.km")

        if current_area < critical_threshold:
            st.error(f"\ud83d\udea8 CRITICAL: Glacier area has dropped below {critical_threshold} sq.km!")
        elif change_rate < -20:
            st.warning("\u26a0\ufe0f ALERT: Rapid glacier shrinkage detected (over 20 sq.km drop).")
        else:
            st.success("\u2705 Glacier area is currently in safe range.")

        if current_area < critical_threshold:
            danger_level = "\ud83d\udd34 High Risk"
        elif current_area < moderate_threshold:
            danger_level = "\ud83d\udfe0 Moderate Risk"
        else:
            danger_level = "\ud83d\udfe2 Low Risk"

        st.markdown(f"### \ud83d\udd16 Danger Level: {danger_level}")

    elif page == "Map Overview":
        st.title("\ud83d\udd98\ufe0f Glacier Region Map Overview")
        st.markdown("Map centered around Gangotri glacier.")
        m = leafmap.Map(center=[30.95, 79.05], zoom=10)
        m.to_streamlit(height=600)

# Chatbot in sidebar
with st.sidebar.expander("\ud83d\udcac Glacier Assistant"):
    st.markdown("Ask me about glaciers, trends, or predictions!")
    user_input = st.text_input("You:", placeholder="e.g., What is glacier retreat?")

    if user_input:
        user_input_lower = user_input.lower()
        if "glacier" in user_input_lower and "retreat" in user_input_lower:
            st.write("\ud83e\uddd1\u200d\ud83e\udd16: Glacier retreat is the process of glaciers shrinking over time due to melting.")
        elif "prediction" in user_input_lower or "2030" in user_input_lower:
            st.write("\ud83e\uddd1\u200d\ud83e\udd16: The glacier area is expected to continue declining through 2030 based on current trends.")
        elif "elevation" in user_input_lower:
            st.write("\ud83e\uddd1\u200d\ud83e\udd16: Elevation trends show variation but generally relate to melting behavior.")
        else:
            st.write("\ud83e\uddd1\u200d\ud83e\udd16: I'm still learning! Try asking about glacier retreat, predictions, or elevation.")
