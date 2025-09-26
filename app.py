import streamlit as st
import pandas as pd
import joblib
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_log_error

st.title("📈 Sales Forecasting App with Evaluation Checks")

# Load trained model
try:
    model = joblib.load("prophet_model.pkl")
    st.success("✅ Loaded trained Prophet model")
except:
    st.error("❌ Could not load trained model. Run train_model.py first.")
    st.stop()

# Upload file
uploaded_file = st.file_uploader("Upload your test CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Test Data")
    st.write(df.head())

    # Check for required column
    if "date" not in df.columns:
        st.error("❌ No 'date' column found.")
        st.stop()

    # Convert date column
    df['date'] = pd.to_datetime(df['date'])

    # Skewness check
    st.subheader("📊 Skewness Check")
    if 'sales' in df.columns:
        skewness = df['sales'].skew()
        st.write(f"Skewness of sales data: {skewness:.3f}")
        if skewness > 1:
            st.warning("⚠️ Highly positively skewed — consider log transformation.")
        elif skewness < -1:
            st.warning("⚠️ Highly negatively skewed — consider transformation.")
        else:
            st.write("Data skewness is acceptable.")
    else:
        st.info("No sales column in test data — skipping skewness check.")

    # Forecast
    st.subheader("📈 Forecast Results")
    future = df[['date']].rename(columns={'date': 'ds'})
    forecast = model.predict(future)
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    # Plot forecast
    st.subheader("Forecast Plot")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Plot components
    st.subheader("Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # Underfit/Overfit check (optional if sales column exists)
    if 'sales' in df.columns:
        st.subheader("📉 Underfit / Overfit Check")
        df_train = df.dropna().copy()
        df_train = df_train.rename(columns={'date': 'ds', 'sales': 'y'})
        split_date = df_train['ds'].iloc[-90]  # last 90 days for validation

        train_set = df_train[df_train['ds'] <= split_date]
        valid_set = df_train[df_train['ds'] > split_date]

        # Train on train set
        m = Prophet()
        m.fit(train_set)

        future_train = m.make_future_dataframe(periods=90)
        forecast_train = m.predict(future_train)

        merged = valid_set.merge(forecast_train[['ds', 'yhat']], on='ds')
        merged['y'] = merged['y'].apply(lambda x: x if x > 0 else 1)
        merged['yhat'] = merged['yhat'].apply(lambda x: x if x > 0 else 1)

        rmsle_val = np.sqrt(mean_squared_log_error(merged['y'], merged['yhat']))
        st.write(f"Validation RMSLE: {rmsle_val:.4f}")

        st.write("Lower RMSLE means better fit. Large difference with training error means overfit or underfit.")

    else:
        st.info("No sales column for underfit/overfit check.")
