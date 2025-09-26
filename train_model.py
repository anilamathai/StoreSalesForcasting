import pandas as pd
from prophet import Prophet
import joblib

# Load training data
df = pd.read_csv("train.csv")

# Prepare data
df['date'] = pd.to_datetime(df['date'])
daily_sales = df.groupby('date')['sales'].sum().reset_index()
daily_sales = daily_sales.rename(columns={'date': 'ds', 'sales': 'y'})

# Train model
model = Prophet()
model.fit(daily_sales)

# Save model
joblib.dump(model, "prophet_model.pkl")
print("✅ Model trained and saved as prophet_model.pkl")
