import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA
import io

# Load your glacier data (Year vs Area CSV)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/yourusername/yourrepo/main/glacier_area.csv"
    df = pd.read_csv(url)
    df = df[['Year', 'Area_sqkm']]
    return df

data = load_data()
years = data['Year'].values.reshape(-1, 1)
areas = data['Area_sqkm'].values

st.title("📈 Glacier Melt Prediction")

# Select forecast range
forecast_year = st.slider("Select Forecast Year", min_value=2025, max_value=2050, value=2035, step=5)

tab1, tab2 = st.tabs(["🔹 Polynomial Regression", "🔸 ARIMA Forecast"])

# -----------------------------
# Tab 1: Polynomial Regression
# -----------------------------
with tab1:
    st.subheader("Polynomial Regression Forecast")

    # Fit polynomial regression
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(years)
    model = LinearRegression()
    model.fit(X_poly, areas)

    # Predict future values
    future_years = np.arange(data['Year'].max() + 1, forecast_year + 1).reshape(-1, 1)
    future_X_poly = poly.transform(future_years)
    future_preds = model.predict(future_X_poly)

    # Combine data
    pred_df_poly = pd.DataFrame({
        'Year': future_years.flatten(),
        'Predicted_Area_sqkm': future_preds
    })

    # Plot
    st.line_chart(pd.concat([
        data.rename(columns={'Area_sqkm': 'Predicted_Area_sqkm'}),
        pred_df_poly
    ]).set_index('Year'))

    # Show R² score
    y_pred_train = model.predict(X_poly)
    r2 = r2_score(areas, y_pred_train)
    st.markdown(f"**R² Score (Training Fit):** {r2:.4f}")

    # Download option
    csv_poly = pred_df_poly.to_csv(index=False)
    st.download_button("📥 Download Polynomial Forecast CSV", csv_poly, "poly_forecast.csv", "text/csv")

# -----------------------------
# Tab 2: ARIMA Forecast
# -----------------------------
with tab2:
    st.subheader("ARIMA Forecast")

    # Fit ARIMA model (you can tune p,d,q)
    arima_model = ARIMA(areas, order=(1, 1, 1))
    arima_result = arima_model.fit()

    n_years_forecast = forecast_year - data['Year'].max()
    forecast_arima = arima_result.forecast(steps=n_years_forecast)

    future_years_arima = np.arange(data['Year'].max() + 1, forecast_year + 1)
    pred_df_arima = pd.DataFrame({
        'Year': future_years_arima,
        'Predicted_Area_sqkm': forecast_arima
    })

    # Plot
    st.line_chart(pd.concat([
        data.rename(columns={'Area_sqkm': 'Predicted_Area_sqkm'}),
        pred_df_arima
    ]).set_index('Year'))

    # Residuals
    residuals = arima_result.resid
    st.markdown("**Residual Analysis (Last 5 residuals):**")
    st.dataframe(residuals.tail())

    # Download
    csv_arima = pred_df_arima.to_csv(index=False)
    st.download_button("📥 Download ARIMA Forecast CSV", csv_arima, "arima_forecast.csv", "text/csv")
