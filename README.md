# 📈 BTC Price Predictor (Streamlit + LSTM)

Aplikacja predykcyjna do prognozowania ceny Bitcoina na kolejne 5 dni, oparta na modelu LSTM (Long Short-Term Memory) i uruchomiona w Streamlit.

## 🔍 Funkcje:

- Automatyczne sprawdzanie i aktualizacja danych BTC (źródło: Binance API)
- Predykcja na kolejne 5 dni na podstawie ostatnich dni zamknięcia
- Interaktywny wykres: 30 dni historycznych + 5 dni prognozy
- Gotowa do hostowania w Streamlit Cloud

## 📁 Struktura projektu:

```
btc-predictor/
├── app.py                # Główna aplikacja Streamlit
├── btcusd_d.csv          # Dane historyczne BTC
├── model.h5              # Wytrenowany model LSTM
├── requirements.txt      # Wymagane biblioteki
└── README.md             # Opis projektu
```

## ▶️ Uruchomienie lokalne

Zainstaluj wymagania:

```bash
pip install -r requirements.txt
```

Uruchom aplikację:

```bash
streamlit run app.py
```

## 🌐 Hosting w chmurze

Projekt można hostować za darmo w Streamlit Cloud: https://streamlit.io/cloud

---

Made with ❤️ by Łukasz Wojciechowski
