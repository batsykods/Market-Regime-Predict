import os
import pandas as pd
import numpy as np


PREDICTIONS_PATH = (
    "data/processed/walk_forward_results.csv"
)

REGIMES_PATH = (
    "data/processed/walk_forward_regimes.csv"
)

OUTPUT_PATH = (
    "data/processed/risk_adjusted_backtest.csv"
)

INITIAL_CAPITAL = 100000
TRANSACTION_COST = 0.001


def calculate_position_size(row):

    prediction = row["Prediction"]
    buy_probability = row["BUY_Probability"]
    volatility = row["Volatility_20"]

    # Only take long positions on BUY
    if prediction != 1:
        return 0.0

    # Confidence-based position sizing
    if buy_probability < 0.55:
        position = 0.0

    elif buy_probability < 0.65:
        position = 0.25

    elif buy_probability < 0.75:
        position = 0.50

    else:
        position = 0.75

    # Reduce exposure during high volatility
    if volatility > 0.30:
        position *= 0.25

    elif volatility > 0.20:
        position *= 0.50

    elif volatility > 0.15:
        position *= 0.75

    return position


def main():

    print("Loading predictions...")

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    print("Loading regime data...")

    regimes = pd.read_csv(
        REGIMES_PATH
    )

    predictions["Date"] = pd.to_datetime(
        predictions["Date"]
    )

    regimes["Date"] = pd.to_datetime(
        regimes["Date"]
    )

    # Get volatility from regime dataset
    volatility_data = regimes[
        [
            "Date",
            "Volatility_20"
        ]
    ]

    # Merge volatility with predictions
    df = predictions.merge(
        volatility_data,
        on="Date",
        how="left"
    )

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    # Remove invalid volatility values
    df["Volatility_20"] = pd.to_numeric(
        df["Volatility_20"],
        errors="coerce"
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=[
            "Volatility_20",
            "Future_Return"
        ]
    ).reset_index(drop=True)

    print(
        f"Rows used: {len(df)}"
    )

    # Calculate desired position
    df["Desired_Position"] = df.apply(
        calculate_position_size,
        axis=1
    )

    # Execute next trading day
    df["Position"] = (
        df["Desired_Position"]
        .shift(1)
        .fillna(0)
    )

    # Market return
    df["Market_Return"] = (
        df["Future_Return"]
    )

    # Strategy return
    df["Strategy_Return"] = (
        df["Position"]
        * df["Market_Return"]
    )

    # Detect trades
    df["Trade"] = (
        df["Position"]
        .diff()
        .abs()
        .fillna(
            df["Position"].abs()
        )
    )

    # Transaction costs
    df["Transaction_Cost"] = (
        df["Trade"]
        * TRANSACTION_COST
    )

    # Return after costs
    df["Strategy_Return_After_Cost"] = (
        df["Strategy_Return"]
        - df["Transaction_Cost"]
    )

    # Strategy equity
    df["Strategy_Equity"] = (
        INITIAL_CAPITAL
        * (
            1
            + df[
                "Strategy_Return_After_Cost"
            ]
        ).cumprod()
    )

    # Buy & Hold equity
    df["Buy_Hold_Equity"] = (
        INITIAL_CAPITAL
        * (
            1
            + df["Market_Return"]
        ).cumprod()
    )

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 50)
    print("RISK-ADJUSTED BACKTEST COMPLETE")
    print("=" * 50)

    print(
        f"Final Strategy Value: "
        f"₹{df['Strategy_Equity'].iloc[-1]:,.2f}"
    )

    print(
        f"Final Buy & Hold Value: "
        f"₹{df['Buy_Hold_Equity'].iloc[-1]:,.2f}"
    )

    print(
        f"Average Position: "
        f"{df['Position'].mean():.2%}"
    )

    print(
        f"Maximum Position: "
        f"{df['Position'].max():.2%}"
    )

    print(
        f"Number of Trades: "
        f"{(df['Trade'] > 0).sum()}"
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()