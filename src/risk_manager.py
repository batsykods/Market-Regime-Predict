import os
import numpy as np
import pandas as pd

PREDICTIONS_PATH = "data/processed/walk_forward_results.csv"
OUTPUT_PATH = "data/processed/risk_adjusted_backtest.csv"
INITIAL_CAPITAL = 100000
TRANSACTION_COST = 0.001
HOLDING_DAYS = 5
PREDICTION_QUANTILE = 0.70
RISK_QUANTILE = 0.65

REGIME_MULTIPLIERS = {0: 1.00, 1: 0.25, 2: 0.70}


def position_size(row, prediction_cutoff, risk_cutoff):
    predicted = float(row["Predicted_Return"])
    volatility = max(float(row["Volatility_20"]), 1e-6)
    regime = int(row["Regime"])
    risk = volatility * np.sqrt(HOLDING_DAYS)

    if predicted < prediction_cutoff:
        return 0.0

    edge = predicted / risk
    if edge < risk_cutoff:
        return 0.0

    if edge < 0.50:
        position = 0.15
    elif edge < 0.75:
        position = 0.30
    elif edge < 1.00:
        position = 0.45
    else:
        position = 0.60

    position *= REGIME_MULTIPLIERS.get(regime, 0.50)

    if volatility > 0.30:
        position *= 0.25
    elif volatility > 0.20:
        position *= 0.50
    elif volatility > 0.15:
        position *= 0.75

    return min(position, 0.60)


def main():
    print("Loading walk-forward 5-day return predictions...")
    df = pd.read_csv(PREDICTIONS_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    required = ["Close", "Future_Return", "Predicted_Return", "Regime", "Volatility_20"]
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).reset_index(drop=True)

    prediction_cutoffs = np.full(len(df), np.nan)
    risk_cutoffs = np.full(len(df), np.nan)
    warmup = max(50, int(len(df) * 0.20))
    risk_ratio = df["Predicted_Return"] / (df["Volatility_20"] * np.sqrt(HOLDING_DAYS))

    for i in range(warmup, len(df)):
        prediction_cutoffs[i] = df.loc[:i - 1, "Predicted_Return"].quantile(PREDICTION_QUANTILE)
        risk_cutoffs[i] = risk_ratio.loc[:i - 1].quantile(RISK_QUANTILE)

    df["Prediction_Cutoff"] = prediction_cutoffs
    df["Risk_Cutoff"] = risk_cutoffs
    df["Desired_Position"] = 0.0

    valid = df["Prediction_Cutoff"].notna() & df["Risk_Cutoff"].notna()
    df.loc[valid, "Desired_Position"] = df.loc[valid].apply(
        lambda r: position_size(r, r["Prediction_Cutoff"], r["Risk_Cutoff"]), axis=1
    )

    positions = np.zeros(len(df), dtype=float)
    active_position = 0.0
    remaining = 0

    for i in range(1, len(df)):
        signal = df.loc[i - 1, "Desired_Position"]
        if signal > 0:
            active_position = max(active_position, signal)
            remaining = HOLDING_DAYS
        elif remaining > 0:
            remaining -= 1
        positions[i] = active_position if remaining > 0 else 0.0
        if remaining == 0:
            active_position = 0.0

    df["Position"] = positions
    df["Market_Return"] = df["Close"].pct_change().fillna(0.0)
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
    print("REGIME-SPECIFIC CALIBRATED 5-DAY BACKTEST COMPLETE")
    print("=" * 50)
    print(f"Prediction quantile: {PREDICTION_QUANTILE:.0%}")
    print(f"Risk quantile: {RISK_QUANTILE:.0%}")
    print(f"Regime multipliers: {REGIME_MULTIPLIERS}")
    print(f"Eligible long signals: {len(active)}")
    print(f"Final Strategy Value: ₹{df['Strategy_Equity'].iloc[-1]:,.2f}")
    print(f"Final Buy & Hold Value: ₹{df['Buy_Hold_Equity'].iloc[-1]:,.2f}")
    print(f"Average Position: {df['Position'].mean():.2%}")
    print(f"Maximum Position: {df['Position'].max():.2%}")
    print(f"Number of Trades: {(df['Trade'] > 0).sum()}")
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
