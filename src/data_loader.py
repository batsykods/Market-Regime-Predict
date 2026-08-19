import os
import pandas as pd
import yfinance as yf


TICKER = "^NSEI"
START_DATE = "2015-01-01"
END_DATE = None

RAW_PATH = "data/raw/nifty50.csv"


def download_market_data():
    os.makedirs("data/raw", exist_ok=True)

    data = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise RuntimeError("No market data was downloaded.")

    # Handle yfinance MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    data.to_csv(RAW_PATH, index=False)

    print(f"Downloaded {len(data)} rows.")
    print(f"Saved to: {RAW_PATH}")

    return data


if __name__ == "__main__":
    download_market_data()