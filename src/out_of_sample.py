import numpy as np
import pandas as pd

INPUT_PATH = "data/processed/walk_forward_results.csv"

# Frozen rules selected before this final evaluation.
PREDICTION_QUANTILE = 0.70
RISK_QUANTILE = 0.65
HOLDING_DAYS = 5
REGIME_MULTIPLIERS = {0: 1.00, 1: 0.25, 2: 0.70}
TRANSACTION_COST = 0.001
INITIAL_CAPITAL = 100000.0


def main():
    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    required = ["Date", "Close", "Future_Return", "Predicted_Return", "Regime", "Volatility_20"]
    df = df.dropna(subset=required).copy()

    split = int(len(df) * 0.70)
    development = df.iloc[:split].copy()
    test = df.iloc[split:].copy().reset_index(drop=True)

    # Thresholds are frozen from development data only.
    pred_cutoff = development["Predicted_Return"].quantile(PREDICTION_QUANTILE)
    risk_ratio_dev = development["Predicted_Return"] / (
        development["Volatility_20"] * np.sqrt(HOLDING_DAYS)
    ).replace(0, np.nan)
    risk_cutoff = risk_ratio_dev.quantile(RISK_QUANTILE)

    test["Risk_Ratio"] = test["Predicted_Return"] / (
        test["Volatility_20"] * np.sqrt(HOLDING_DAYS)
    ).replace(0, np.nan)
    test["Signal"] = (
        (test["Predicted_Return"] >= pred_cutoff)
        & (test["Risk_Ratio"] >= risk_cutoff)
    )

    def size(row):
        if not row["Signal"]:
            return 0.0
        edge = row["Risk_Ratio"]
        if edge < 0.50:
            p = 0.15
        elif edge < 0.75:
            p = 0.30
        elif edge < 1.00:
            p = 0.45
        else:
            p = 0.60
        p *= REGIME_MULTIPLIERS.get(int(row["Regime"]), 0.50)
        return min(p, 0.60)

    test["Desired_Position"] = test.apply(size, axis=1)
    positions = np.zeros(len(test))
    active = 0.0
    remaining = 0

    for i in range(1, len(test)):
        signal = test.loc[i - 1, "Desired_Position"]
        if signal > 0:
            active = max(active, signal)
            remaining = HOLDING_DAYS
        elif remaining > 0:
            remaining -= 1
        positions[i] = active if remaining > 0 else 0.0
        if remaining == 0:
            active = 0.0

    test["Position"] = positions
    test["Market_Return"] = test["Close"].pct_change().fillna(0.0)
    test["Strategy_Return"] = test["Position"] * test["Market_Return"]
    test["Trade"] = test["Position"].diff().abs().fillna(test["Position"].abs())
    test["Net_Return"] = test["Strategy_Return"] - test["Trade"] * TRANSACTION_COST
    equity = INITIAL_CAPITAL * (1 + test["Net_Return"]).cumprod()
    buy_hold = INITIAL_CAPITAL * (1 + test["Market_Return"]).cumprod()

    print("\n" + "=" * 60)
    print("FROZEN OUT-OF-SAMPLE VALIDATION")
    print("=" * 60)
    print(f"Development: {development['Date'].iloc[0].date()} -> {development['Date'].iloc[-1].date()}")
    print(f"Test:        {test['Date'].iloc[0].date()} -> {test['Date'].iloc[-1].date()}")
    print(f"Frozen prediction cutoff: {pred_cutoff:.6f}")
    print(f"Frozen risk cutoff: {risk_cutoff:.6f}")
    print(f"Signals: {int(test['Signal'].sum())}")
    print(f"Final strategy value: ₹{equity.iloc[-1]:,.2f}")
    print(f"Final buy & hold value: ₹{buy_hold.iloc[-1]:,.2f}")
    print(f"Strategy return: {(equity.iloc[-1] / INITIAL_CAPITAL - 1):.4%}")
    print(f"Buy & hold return: {(buy_hold.iloc[-1] / INITIAL_CAPITAL - 1):.4%}")
    print(f"Average exposure: {test['Position'].mean():.2%}")
    print(f"Trades: {(test['Trade'] > 0).sum()}")
    print(f"Direction accuracy: {(np.sign(test['Future_Return']) == np.sign(test['Predicted_Return'])).mean():.4f}")


if __name__ == "__main__":
    main()
