import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import datetime

# Load glacier area dataset from GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/your-github/glacier-data/main/gangotri_area.csv"
    df = pd.read_csv(url)
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df.set_index('Year', inplace=True)
    return df

data = load_data()

# Title and forecast range selection
st.title("📉 Glacier Melt Forecast - Gangotri Glacier")
st.markdown("Analyze and predict glacier melt using Polynomial Regression and ARIMA.")

forecast_year = st.slider("Select forecast horizon:", 2025, 2050, 2035)

# Tabs for prediction models
tabs = st.tabs(["📈 Polynomial Regression", "🔮 ARIMA Forecast"])

# ----------------- Polynomial Regression -------------------
with tabs[0]:
    st.subheader("Polynomial Regression Prediction")

    df_poly = data.reset_index()
    df_poly['Year'] = df_poly['Year'].dt.year
    X = df_poly[['Year']]
    y = df_poly['Area_km2']

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    future_years = np.arange(df_poly['Year'].max() + 1, forecast_year + 1).reshape(-1, 1)
    future_poly = poly.transform(future_years)
    future_preds = model.predict(future_poly)

    r2 = r2_score(y, model.predict(X_poly))
    residuals = y - model.predict(X_poly)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(X, y, label="Observed", marker='o')
    ax.plot(future_years, future_preds, label="Predicted", linestyle='--', color='red')
    ax.set_xlabel("Year")
    ax.set_ylabel("Glacier Area (km²)")
    ax.set_title("Polynomial Regression Forecast")
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"**R² Score:** {r2:.4f}")
    st.markdown(f"**Mean Squared Error:** {mean_squared_error(y, model.predict(X_poly)):.4f}")

    # Alert system
    loss = y.iloc[-1] - future_preds[-1]
    if loss > 10:
        st.error(f"⚠️ Significant glacier loss predicted: {loss:.2f} km² by {forecast_year}")
    else:
        st.success(f"✅ Moderate glacier loss: {loss:.2f} km²")

    # Export
    pred_df = pd.DataFrame({
        "Year": future_years.flatten(),
        "Predicted_Area_km2": future_preds
    })
    csv = pred_df.to_csv(index=False).encode()
    st.download_button("📥 Download Polynomial Prediction CSV", csv, "polynomial_forecast.csv", "text/csv")

# ----------------- ARIMA Forecast -------------------
with tabs[1]:
    st.subheader("ARIMA Forecast")

    df_arima = data.copy()
    df_arima.index = df_arima.index.year
    model = ARIMA(df_arima['Area_km2'], order=(2, 1, 2))
    fitted_model = model.fit()

    steps = forecast_year - df_arima.index.max()
    forecast = fitted_model.forecast(steps=steps)
    forecast_years = np.arange(df_arima.index.max() + 1, forecast_year + 1)

    # Plot
    fig2, ax2 = plt.subplots()
    ax2.plot(df_arima.index, df_arima['Area_km2'], label="Observed", marker='o')
    ax2.plot(forecast_years, forecast, label="Forecast", linestyle='--', color='green')
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Glacier Area (km²)")
    ax2.set_title("ARIMA Forecast")
    ax2.legend()
    st.pyplot(fig2)

    # Residuals
    arima_fitted = fitted_model.fittedvalues
    residuals = df_arima['Area_km2'].iloc[1:] - arima_fitted
    st.markdown(f"**Mean Squared Error (ARIMA):** {mean_squared_error(df_arima['Area_km2'].iloc[1:], arima_fitted):.4f}")

    # Alert
    arima_loss = df_arima['Area_km2'].iloc[-1] - forecast.iloc[-1]
    if arima_loss > 10:
        st.error(f"⚠️ ARIMA predicts a glacier loss of {arima_loss:.2f} km² by {forecast_year}")
    else:
        st.success(f"✅ ARIMA projects stable glacier area loss: {arima_loss:.2f} km²")

    # Export
    arima_df = pd.DataFrame({
        "Year": forecast_years,
        "Forecasted_Area_km2": forecast.values
    })
    csv_arima = arima_df.to_csv(index=False).encode()
    st.download_button("📥 Download ARIMA Forecast CSV", csv_arima, "arima_forecast.csv", "text/csv")

# Footer clock
st.markdown("---")
st.markdown(f"🕒 Current IST: {datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}")
