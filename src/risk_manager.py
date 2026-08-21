import os
import numpy as np
import pandas as pd

PREDICTIONS_PATH = "data/processed/walk_forward_results.csv"
OUTPUT_PATH = "data/processed/risk_adjusted_backtest.csv"

INITIAL_CAPITAL = 100000
TRANSACTION_COST = 0.001


def position_size(row):
    predicted = float(row["Predicted_Return"])
    volatility = max(float(row["Volatility_20"]), 1e-6)
    regime = int(row["Regime"])

    # Trade only when predicted return is positive and large enough
    # to compensate for estimated one-day noise and transaction costs.
    threshold = max(0.0015, 0.20 * volatility)
    if predicted <= threshold:
        return 0.0

    # Signal strength relative to current volatility.
    edge = predicted / volatility
    if edge < 0.20:
        position = 0.10
    elif edge < 0.35:
        position = 0.20
    elif edge < 0.50:
        position = 0.30
    else:
        position = 0.40

    # Regime is a risk modifier, not a prediction override.
    if regime == 0:
        position *= 0.50
    elif regime == 2:
        position *= 0.75

    if volatility > 0.30:
        position *= 0.25
    elif volatility > 0.20:
        position *= 0.50
    elif volatility > 0.15:
        position *= 0.75

    return min(position, 0.40)


def main():
    print("Loading walk-forward return predictions...")
    df = pd.read_csv(PREDICTIONS_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    required = ["Future_Return", "Predicted_Return", "Regime", "Volatility_20"]
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).reset_index(drop=True)
    print(f"Rows used: {len(df)}")

    df["Desired_Position"] = df.apply(position_size, axis=1)

    # Prediction made at today's close is executed for the next session.
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
    print("RISK-ADJUSTED RETURN BACKTEST COMPLETE")
    print("=" * 50)
    print(f"Eligible long signals: {len(active)}")
    print(f"Final Strategy Value: ₹{df['Strategy_Equity'].iloc[-1]:,.2f}")
    print(f"Final Buy & Hold Value: ₹{df['Buy_Hold_Equity'].iloc[-1]:,.2f}")
    print(f"Average Position: {df['Position'].mean():.2%}")
    print(f"Maximum Position: {df['Position'].max():.2%}")
    print(f"Number of Trades: {(df['Trade'] > 0).sum()}")
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
