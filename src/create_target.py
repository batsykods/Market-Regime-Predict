import pandas as pd


INPUT_PATH = "data/processed/nifty50_regimes.csv"
OUTPUT_PATH = "data/processed/nifty50_signal_data.csv"


BUY_THRESHOLD = 0.005
SELL_THRESHOLD = -0.005


def main():

    df = pd.read_csv(INPUT_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    # Tomorrow's return
    df["Future_Return"] = (
        df["Close"].shift(-1) / df["Close"]
    ) - 1

    # 0 = HOLD
    # 1 = BUY
    # 2 = SELL
    df["Target"] = 0

    df.loc[
        df["Future_Return"] > BUY_THRESHOLD,
        "Target"
    ] = 1

    df.loc[
        df["Future_Return"] < SELL_THRESHOLD,
        "Target"
    ] = 2

    # Remove final row because tomorrow's
    # return does not exist.
    df = df.dropna(
        subset=["Future_Return"]
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("Signal dataset created.")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nTarget distribution:")
    print(df["Target"].value_counts())

    print("\nTarget percentages:")
    print(
        df["Target"]
        .value_counts(normalize=True)
        .sort_index()
    )


if __name__ == "__main__":
    main()