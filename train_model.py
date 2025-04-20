import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Wczytaj dane
df = pd.read_csv("btcusd_d.csv")
prices = df["Zamkniecie"].values.reshape(-1, 1)

# Normalizacja
scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices)

# Sekwencje
window = 5
X, y = [], []
for i in range(window, len(scaled)):
    X.append(scaled[i-window:i, 0])
    y.append(scaled[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Model
model = Sequential([
    LSTM(50, input_shape=(window, 1)),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=20, batch_size=32)

# Zapis
model.save("model.h5")
print("✅ Zapisano model.h5")
