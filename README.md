# fxdashboard

Streamlit FX analytics terminal for major currency pairs: live-style quotes, technical charts, correlations, currency strength, and forex news with sentiment.

## Features

- Overview tickers for EUR/USD, GBP/USD, USD/JPY, USD/CHF, USD/CAD, AUD/USD, NZD/USD (Yahoo Finance FX symbols)
- Per-pair trend / RSI / MACD / volatility signal labels from technical indicators
- Charts tab: candlesticks with EMA 20/50/200, RSI, and MACD (Plotly)
- Correlations tab: return correlation matrix across major pairs
- Heat map tab: simple currency strength bars derived from recent returns
- Annualized volatility progress display for the selected pair
- Forex news categorized as breaking, market, and analysis (NewsAPI + TextBlob sentiment)

## Requirements

See `requirements.txt` (streamlit, pandas, numpy, yfinance, plotly, textblob, newsapi-python, ta, python-dateutil, pytz).

## Setup

1. Clone the repo and create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Obtain a NewsAPI key. The app currently constructs `NewsAggregator` with a key in `fxdashboard.py`; move it to Streamlit secrets or an environment variable before sharing or deploying.

## Usage

```bash
streamlit run fxdashboard.py
```

The default interval used in `main()` is `1M` (1-minute bars). Use the pair selector and Charts / Correlations / Heat Map tabs for deeper views.

## Notes

- Data comes from Yahoo Finance FX tickers (e.g. `EURUSD=X`).
- Do not commit live NewsAPI keys; rotate any key that was previously hardcoded in source.
