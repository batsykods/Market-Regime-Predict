import os
import numpy as np
import pandas as pd


RAW_PATH = "data/raw/nifty50.csv"
PROCESSED_PATH = "data/processed/nifty50_features.csv"


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["Return"] = df["Close"].pct_change()

    df["Log_Return"] = np.log(
        df["Close"] / df["Close"].shift(1)
    )

    df["Volatility_20"] = (
        df["Log_Return"]
        .rolling(window=20)
        .std()
        * np.sqrt(252)
    )

    df["SMA_20"] = (
        df["Close"]
        .rolling(window=20)
        .mean()
    )

    df["SMA_50"] = (
        df["Close"]
        .rolling(window=50)
        .mean()
    )

    df["Momentum_20"] = (
        df["Close"] /
        df["Close"].shift(20)
        - 1
    )

    high_low = df["High"] - df["Low"]

    high_close = (
        df["High"] -
        df["Close"].shift(1)
    ).abs()

    low_close = (
        df["Low"] -
        df["Close"].shift(1)
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    df["ATR_14"] = (
        true_range
        .rolling(14)
        .mean()
    )



    df["Volume_Change"] = (
        df["Volume"].pct_change()
    )

    df["Drawdown"] = (
        df["Close"] /
        df["Close"].cummax()
        - 1
    )

    return df


def main():

    df = pd.read_csv(RAW_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    df = create_features(df)

    # Remove rows where rolling calculations
    # cannot yet be computed.
    df = df.dropna().reset_index(drop=True)

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(PROCESSED_PATH, index=False)

    print(f"Created {len(df)} processed rows.")
    print(f"Saved to: {PROCESSED_PATH}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nLatest data:")
    print(df.tail())


if __name__ == "__main__":
    main()