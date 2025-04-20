# 📈 BTC Price Predictor (Streamlit + LSTM)

A simple yet powerful Bitcoin price forecasting app built with LSTM (Long Short-Term Memory) and deployed using Streamlit Cloud. The app predicts the BTC price for the next 5 days based on historical closing prices.

🔗 **Live App**: [btc-predictor-the28s.streamlit.app](https://btc-predictor-the28s.streamlit.app)

---

## 🔍 Features

- ✅ Automatically checks for new BTC data from Binance (fallback: CoinGecko)
- 📅 Updates `btcusd_d.csv` with the latest available closing price
- 🤖 Forecasts the next 5 days using a pre-trained LSTM model
- 📊 Displays the last 30 days + 5-day forecast in an interactive chart
- ☁️ Hosted live on Streamlit Cloud
- 💾 Works even without daily manual launches — backfills missing days

---

## 🗂️ Project Structure

```
btc-predictor/
├── app.py              # Main Streamlit app
├── btcusd_d.csv        # Historical BTC data (auto-updated)
├── model.h5            # Pre-trained LSTM model
├── requirements.txt    # List of Python dependencies
├── train_model.py      # Script to train the model (optional)
├── utils.py            # (Optional) Utilities module
└── README.md           # This file
```

---

## ▶️ Run Locally

1. Clone the repository:

```bash
git clone https://github.com/yourusername/btc-predictor.git
cd btc-predictor
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 🚀 Deployed on Streamlit Cloud

This app is fully hosted and automatically runs online at:

👉 https://btc-predictor-the28s.streamlit.app

No need to run anything locally if you just want to use it or share it!

---

## 🛠️ Technology Stack

- Python
- Streamlit
- TensorFlow / Keras (LSTM)
- Pandas & NumPy
- Matplotlib
- Binance & CoinGecko APIs

---

## 📌 Note

If Binance API fails (e.g., due to region blocks or rate limits), the app will fallback to CoinGecko to get the latest BTC closing price.

---

## 👨‍💻 Author

**Łukasz Wojciechowski**  
[the28s.com](https://the28s.com)  
Made with ❤️ in Vienna
