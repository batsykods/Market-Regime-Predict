import yfinance as yf
import pandas as pd
import numpy as np


def get_live_data():

    df = yf.Ticker("^NSEI").history(
        period="6mo",
        interval="1d"
    )

    df = df.reset_index()

    df = df.rename(
        columns={
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume"
        }
    )

    # Returns
    df["Log_Return"] = np.log(
        df["Close"] / df["Close"].shift(1)
    )

    # Volatility
    df["Volatility_20"] = (
        df["Log_Return"]
        .rolling(20)
        .std()
    )

    # Moving averages
    df["SMA_20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    df["SMA_50"] = (
        df["Close"]
        .rolling(50)
        .mean()
    )

    # Momentum
    df["Momentum_20"] = (
        df["Close"]
        / df["Close"].shift(20)
        - 1
    )

    # Volume change
    df["Volume_Change"] = (
        df["Volume"]
        .pct_change()
    )

    # Drawdown
    rolling_max = df["Close"].cummax()

    df["Drawdown"] = (
        df["Close"]
        / rolling_max
        - 1
    )

    # Remove invalid values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna().reset_index(
        drop=True
    )

    return df


if __name__ == "__main__":

    df = get_live_data()

    print("\nLatest live features:")
    print(
        df[
            [
                "Date",
                "Close",
                "Log_Return",
                "Volatility_20",
                "SMA_20",
                "SMA_50",
                "Momentum_20",
                "Volume_Change",
                "Drawdown"
            ]
        ].tail(1).to_string(
            index=False
        )
    )