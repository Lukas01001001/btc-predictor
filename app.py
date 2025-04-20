import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import requests
import matplotlib.pyplot as plt

# === SETTINGS ===
DATA_CSV = "btcusd_d.csv"
MODEL_FILE = "model.h5"
WINDOW_SIZE = 5
FUTURE_DAYS = 5

# === HELPER FUNCTIONS ===
def load_data():
    df = pd.read_csv(DATA_CSV)
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df = df.dropna(subset=['Data'])  # usuwanie błędnych wierszy
    return df

def save_data(df):
    df.to_csv(DATA_CSV, index=False)

def fetch_latest_data(last_date):
    today = datetime.now().date()
    if last_date >= today:
        return []

    # Try Binance API
    binance_url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d"
    response = requests.get(binance_url)

    if response.status_code == 200:
        data = response.json()
        rows = []
        for row in data:
            ts = datetime.fromtimestamp(row[0] / 1000).date()
            if ts > last_date:
                close = float(row[4])
                rows.append({'Data': ts, 'Zamkniecie': close})
        if rows:
            return rows
        else:
            st.warning("Binance API returned data, but no new dates were found.")
    else:
        st.warning(f"⚠️ Binance API error: {response.status_code}")

    # Fallback: Try CoinGecko API
    coingecko_url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1&interval=daily"
    cg_response = requests.get(coingecko_url)

    if cg_response.status_code == 200:
        cg_data = cg_response.json()
        prices = cg_data.get("prices", [])
        if prices:
            ts = datetime.fromtimestamp(prices[-1][0] / 1000).date()
            if ts > last_date:
                close = prices[-1][1]
                return [{'Data': ts, 'Zamkniecie': close}]
            else:
                st.info("✅ CoinGecko API responded, but the latest data is already present.")
        else:
            st.warning("⚠️ CoinGecko API returned no price data.")
    else:
        st.error(f"❌ CoinGecko API error: {cg_response.status_code}")

    return []

def predict_next_days(prices, model, scaler):
    scaled = scaler.transform(prices.reshape(-1, 1))
    last_sequence = scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
    future_predictions = []

    for _ in range(FUTURE_DAYS):
        next_scaled = model.predict(last_sequence)[0][0]
        future_predictions.append(next_scaled)
        last_sequence = np.append(last_sequence[:, 1:, :], [[[next_scaled]]], axis=1)

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# === STREAMLIT APP ===
st.set_page_config(page_title="Bitcoin Price Forecast", layout="centered")
st.title("📈 Bitcoin Price Forecast (LSTM)")

# Load model
model = load_model(MODEL_FILE)

# Load data
df = load_data()

# Show data range
st.caption(f"🗓️ Data range: {df['Data'].min().date()} → {df['Data'].max().date()} ({len(df)} days total)")

# Check and update data
last_date = df['Data'].max().date()
new_rows = fetch_latest_data(last_date)

if new_rows:
    st.info(f"📅 Data updated with {len(new_rows)} new day(s).")
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    save_data(df)
else:
    st.success("✅ Data is up to date.")

# Scale data
prices = df['Zamkniecie'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaler.fit(prices)

# Predict
predicted = predict_next_days(prices, model, scaler)

# Display forecast
st.subheader("🔮 Forecast for the next 5 days:")
for i, val in enumerate(predicted.flatten(), 1):
    st.write(f"Day {i}: **${val:,.2f}**")

# Chart
st.subheader("📊 Chart")
st.markdown("📡 *Latest price fetched from: Binance (fallback: CoinGecko)*")

last_prices = prices[-30:].flatten()
forecast_prices = predicted.flatten()
combined = np.concatenate([last_prices, forecast_prices])
x_labels = [f"-{29 - i}" for i in range(30)] + [f"+{i+1}" for i in range(FUTURE_DAYS)]

fig, ax = plt.subplots()
ax.plot(x_labels, combined, marker='o')
ax.axvline(x=29.5, color='gray', linestyle='--')
ax.set_title("BTC Price – Last 30 Days and 5-Day Forecast")
ax.set_xlabel("Days")
ax.set_ylabel("Price (USD)")
plt.xticks(rotation=45)
st.pyplot(fig)
