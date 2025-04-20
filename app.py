
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import requests
import matplotlib.pyplot as plt

# === USTAWIENIA ===
DATA_CSV = "btcusd_d.csv"
MODEL_FILE = "model.h5"
WINDOW_SIZE = 5
FUTURE_DAYS = 5

# === FUNKCJE POMOCNICZE ===
def load_data():
    df = pd.read_csv(DATA_CSV)
    df['Data'] = pd.to_datetime(df['Data'])
    return df

def save_data(df):
    df.to_csv(DATA_CSV, index=False)

def fetch_latest_data(last_date):
    today = datetime.now().date()
    if last_date >= today:
        return []

    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Błąd pobierania danych z Binance API")
        return []

    data = response.json()
    rows = []
    for row in data:
        ts = datetime.fromtimestamp(row[0] / 1000).date()
        if ts > last_date:
            close = float(row[4])
            rows.append({'Data': ts, 'Zamkniecie': close})
    return rows

def predict_next_days(prices, model, scaler):
    scaled = scaler.transform(prices.reshape(-1, 1))
    last_sequence = scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
    future_predictions = []

    for _ in range(FUTURE_DAYS):
        next_scaled = model.predict(last_sequence)[0][0]
        future_predictions.append(next_scaled)
        last_sequence = np.append(last_sequence[:, 1:, :], [[[next_scaled]]], axis=1)

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# === APLIKACJA STREAMLIT ===
st.set_page_config(page_title="BTC Price Forecast", layout="centered")
st.title("📈 Prognoza ceny Bitcoina (LSTM)")

# Wczytanie modelu
model = load_model(MODEL_FILE)

# Wczytanie danych
df = load_data()

# Sprawdzenie i aktualizacja danych
last_date = df['Data'].max().date()
new_rows = fetch_latest_data(last_date)

if new_rows:
    st.info(f"Zaktualizowano dane o {len(new_rows)} dni.")
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    save_data(df)
else:
    st.success("Dane są aktualne.")

# Skalowanie danych
prices = df['Zamkniecie'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaler.fit(prices)

# Predykcja
predicted = predict_next_days(prices, model, scaler)

# Wyświetlenie prognoz
st.subheader("🔮 Prognoza na kolejne 5 dni:")
for i, val in enumerate(predicted.flatten(), 1):
    st.write(f"Dzień {i}: **${val:,.2f}**")

# Wykres
st.subheader("📊 Wykres")
last_prices = prices[-30:].flatten()
forecast_prices = predicted.flatten()
combined = np.concatenate([last_prices, forecast_prices])
x_labels = [f"-{29 - i}" for i in range(30)] + [f"+{i+1}" for i in range(FUTURE_DAYS)]

fig, ax = plt.subplots()
ax.plot(x_labels, combined, marker='o')
ax.axvline(x=29.5, color='gray', linestyle='--')
ax.set_title("Cena BTC – ostatnie 30 dni i prognoza 5 dni")
ax.set_xlabel("Dni")
ax.set_ylabel("Cena (USD)")
plt.xticks(rotation=45)
st.pyplot(fig)
