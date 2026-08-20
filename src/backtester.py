import pandas as pd
import numpy as np


INPUT_PATH = "data/processed/nifty50_signal_data.csv"

INITIAL_CAPITAL = 100000
TRANSACTION_COST = 0.001


def load_data():

    df = pd.read_csv(INPUT_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    return df


def generate_signals(df, model, features):

    X = df[features]

    predictions = model.predict(X)

    df["Signal"] = predictions

    return df


def create_positions(df):

    # Signal generated after today's close.
    # Execute from the next trading day.
    df["Position"] = (
        df["Signal"]
        .shift(1)
        .fillna(0)
    )

    # BUY = 1
    # HOLD = 0
    # SELL = 2

    # Convert signals into actual exposure.
    #
    # BUY  -> 1
    # HOLD -> previous position
    # SELL -> 0

    position = 0
    positions = []

    for signal in df["Signal"]:

        if signal == 1:
            position = 1

        elif signal == 2:
            position = 0

        positions.append(position)

    df["Position"] = (
        pd.Series(positions)
        .shift(1)
        .fillna(0)
    )

    return df


def calculate_returns(df):

    df["Market_Return"] = (
        df["Close"]
        .pct_change().fillna(0)
    )

    df["Strategy_Return"] = (
        df["Position"]
        * df["Market_Return"]
    )

    # Detect position changes.
    df["Trade"] = (
        df["Position"]
        .diff()
        .abs()
        .fillna(0)
    )

    # Transaction costs.
    df["Strategy_Return_After_Cost"] = (
        df["Strategy_Return"]
        - df["Trade"] * TRANSACTION_COST
    )

    return df


def calculate_equity(df):

    df["Strategy_Equity"] = (
        INITIAL_CAPITAL
        * (
            1 + df["Strategy_Return_After_Cost"]
        ).cumprod()
    )

    df["Buy_Hold_Equity"] = (
        INITIAL_CAPITAL
        * (
            1 + df["Market_Return"].fillna(0)
        ).cumprod()
    )

    return df


def main():

    import joblib

    model = joblib.load(
        "data/models/xgboost_signal_model.pkl"
    )

    features = [
        "Log_Return",
        "Volatility_20",
        "SMA_20",
        "SMA_50",
        "Momentum_20",
        "Volume_Change",
        "Drawdown",
        "Regime"
    ]

    df = load_data()

    df = generate_signals(
        df,
        model,
        features
    )

    df = create_positions(df)

    df = calculate_returns(df)

    df = calculate_equity(df)

    output = (
        "data/processed/"
        "backtest_results.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print(
        f"Backtest saved to: {output}"
    )

    print(
        f"\nInitial Capital: "
        f"₹{INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Strategy Final Value: "
        f"₹{df['Strategy_Equity'].iloc[-1]:,.2f}"
    )

    print(
        f"Buy & Hold Final Value: "
        f"₹{df['Buy_Hold_Equity'].iloc[-1]:,.2f}"
    )


if __name__ == "__main__":
    main()