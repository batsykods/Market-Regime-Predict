import os
import numpy as np
import pandas as pd

PREDICTIONS_PATH = "data/processed/walk_forward_results.csv"
REGIMES_PATH = "data/processed/walk_forward_regimes.csv"
OUTPUT_PATH = "data/processed/risk_adjusted_backtest.csv"

INITIAL_CAPITAL = 100000
TRANSACTION_COST = 0.001


def calculate_position_size(row):
    buy = float(row["BUY_Probability"])
    hold = float(row["HOLD_Probability"])
    sell = float(row["SELL_Probability"])
    regime = int(row["Regime"])
    volatility = float(row["Volatility_20"])
    prediction = int(row["Prediction"])

    # Only take the model's BUY class; no discretionary label override.
    if prediction != 1:
        return 0.0

    # The walk-forward model never reaches 0.55 on most observations.
    # Use a minimum probability plus a small edge over the runner-up.
    runner_up = max(hold, sell)
    edge = buy - runner_up
    if buy < 0.40 or edge < 0.03:
        return 0.0

    # Confidence-based base exposure.
    if buy >= 0.50:
        position = 0.50
    elif buy >= 0.45:
        position = 0.35
    else:
        position = 0.20

    # Regime modifies, rather than vetoes, exposure.
    if regime == 0:
        position *= 0.50
    elif regime == 2:
        position *= 0.75

    # Volatility cap.
    if volatility > 0.30:
        position *= 0.25
    elif volatility > 0.20:
        position *= 0.50
    elif volatility > 0.15:
        position *= 0.75

    return min(position, 0.50)


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
        "Future_Return", "Prediction", "HOLD_Probability",
        "BUY_Probability", "SELL_Probability", "Regime",
        "Volatility_20",
    ]
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required).reset_index(drop=True)

    print(f"Rows used: {len(df)}")

    df["Desired_Position"] = df.apply(calculate_position_size, axis=1)
    df["Position"] = df["Desired_Position"].shift(1).fillna(0.0)

    df["Market_Return"] = df["Future_Return"]
    df["Strategy_Return"] = df["Position"] * df["Market_Return"]
    df["Trade"] = df["Position"].diff().abs().fillna(df["Position"].abs())
    df["Transaction_Cost"] = df["Trade"] * TRANSACTION_COST
    df["Strategy_Return_After_Cost"] = df["Strategy_Return"] - df["Transaction_Cost"]

    df["Strategy_Equity"] = INITIAL_CAPITAL * (1 + df["Strategy_Return_After_Cost"]).cumprod()
    df["Buy_Hold_Equity"] = INITIAL_CAPITAL * (1 + df["Market_Return"]).cumprod()

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    active = df[df["Desired_Position"] > 0]
    print("\n" + "=" * 50)
    print("RISK-ADJUSTED BACKTEST COMPLETE")
    print("=" * 50)
    print(f"Eligible BUY signals: {len(active)}")
    print(f"Final Strategy Value: ₹{df['Strategy_Equity'].iloc[-1]:,.2f}")
    print(f"Final Buy & Hold Value: ₹{df['Buy_Hold_Equity'].iloc[-1]:,.2f}")
    print(f"Average Position: {df['Position'].mean():.2%}")
    print(f"Maximum Position: {df['Position'].max():.2%}")
    print(f"Number of Trades: {(df['Trade'] > 0).sum()}")
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
