import os
import numpy as np
import pandas as pd

PREDICTIONS_PATH = "data/processed/walk_forward_results.csv"
REGIMES_PATH = "data/processed/walk_forward_regimes.csv"
OUTPUT_PATH = "data/processed/risk_adjusted_backtest.csv"

INITIAL_CAPITAL = 100000
TRANSACTION_COST = 0.001


def calculate_position_size(row):
    buy_probability = float(row["BUY_Probability"])
    sell_probability = float(row["SELL_Probability"])
    hold_probability = float(row["HOLD_Probability"])
    regime = int(row["Regime"])
    volatility = float(row["Volatility_20"])

    # Regime 0 is the historically weakest regime in this project.
    if regime == 0:
        return 0.0

    # Require BUY to be the strongest class and have a meaningful margin.
    strongest = max(buy_probability, hold_probability, sell_probability)
    if buy_probability != strongest:
        return 0.0

    confidence_margin = buy_probability - max(
        hold_probability,
        sell_probability,
    )

    if buy_probability < 0.55 or confidence_margin < 0.10:
        return 0.0

    if buy_probability < 0.65:
        position = 0.25
    elif buy_probability < 0.75:
        position = 0.50
    else:
        position = 0.75

    # Volatility is annualised in the feature pipeline.
    if volatility > 0.30:
        position *= 0.25
    elif volatility > 0.20:
        position *= 0.50
    elif volatility > 0.15:
        position *= 0.75

    return position


def main():
    print("Loading predictions...")
    predictions = pd.read_csv(PREDICTIONS_PATH)

    print("Loading regime data...")
    regimes = pd.read_csv(REGIMES_PATH)

    predictions["Date"] = pd.to_datetime(predictions["Date"])
    regimes["Date"] = pd.to_datetime(regimes["Date"])

    volatility_data = regimes[["Date", "Volatility_20"]]

    df = predictions.merge(
        volatility_data,
        on="Date",
        how="left",
        validate="one_to_one",
    )

    df = df.sort_values("Date").reset_index(drop=True)

    required = [
        "Future_Return",
        "Prediction",
        "HOLD_Probability",
        "BUY_Probability",
        "SELL_Probability",
        "Regime",
        "Volatility_20",
    ]

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required).reset_index(drop=True)

    print(f"Rows used: {len(df)}")

    df["Desired_Position"] = df.apply(
        calculate_position_size,
        axis=1,
    )

    # A signal generated at today's close is executed for the next session.
    df["Position"] = (
        df["Desired_Position"].shift(1).fillna(0.0)
    )

    df["Market_Return"] = df["Future_Return"]
    df["Strategy_Return"] = (
        df["Position"] * df["Market_Return"]
    )

    df["Trade"] = (
        df["Position"].diff().abs().fillna(df["Position"].abs())
    )

    df["Transaction_Cost"] = (
        df["Trade"] * TRANSACTION_COST
    )

    df["Strategy_Return_After_Cost"] = (
        df["Strategy_Return"] - df["Transaction_Cost"]
    )

    df["Strategy_Equity"] = (
        INITIAL_CAPITAL
        * (1 + df["Strategy_Return_After_Cost"]).cumprod()
    )

    df["Buy_Hold_Equity"] = (
        INITIAL_CAPITAL
        * (1 + df["Market_Return"]).cumprod()
    )

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 50)
    print("RISK-ADJUSTED BACKTEST COMPLETE")
    print("=" * 50)
    print(f"Final Strategy Value: ₹{df['Strategy_Equity'].iloc[-1]:,.2f}")
    print(f"Final Buy & Hold Value: ₹{df['Buy_Hold_Equity'].iloc[-1]:,.2f}")
    print(f"Average Position: {df['Position'].mean():.2%}")
    print(f"Maximum Position: {df['Position'].max():.2%}")
    print(f"Number of Trades: {(df['Trade'] > 0).sum()}")
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
