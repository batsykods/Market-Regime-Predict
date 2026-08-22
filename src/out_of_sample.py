import numpy as np
import pandas as pd

INPUT_PATH = "data/processed/walk_forward_results.csv"
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

    pred_cutoff = development["Predicted_Return"].quantile(PREDICTION_QUANTILE)
    dev_risk = development["Predicted_Return"] / (development["Volatility_20"] * np.sqrt(HOLDING_DAYS)).replace(0, np.nan)
    risk_cutoff = dev_risk.quantile(RISK_QUANTILE)
    test["Risk_Ratio"] = test["Predicted_Return"] / (test["Volatility_20"] * np.sqrt(HOLDING_DAYS)).replace(0, np.nan)
    test["Signal"] = (test["Predicted_Return"] >= pred_cutoff) & (test["Risk_Ratio"] >= risk_cutoff)

    def size(r):
        if not r["Signal"]:
            return 0.0
        e = r["Risk_Ratio"]
        p = 0.15 if e < 0.50 else 0.30 if e < 0.75 else 0.45 if e < 1.00 else 0.60
        p *= REGIME_MULTIPLIERS.get(int(r["Regime"]), 0.50)
        v = r["Volatility_20"]
        p *= 0.25 if v > 0.30 else 0.50 if v > 0.20 else 0.75 if v > 0.15 else 1.0
        return min(p, 0.60)

    test["Desired_Position"] = test.apply(size, axis=1)
    positions = np.zeros(len(test)); active = 0.0; remaining = 0
    for i in range(1, len(test)):
        signal = test.loc[i - 1, "Desired_Position"]
        if signal > 0:
            active = max(active, signal); remaining = HOLDING_DAYS
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
    curve = (1 + test["Net_Return"]).cumprod()
    bh_curve = (1 + test["Market_Return"]).cumprod()
    r = test["Net_Return"]
    std = r.std(ddof=1); downside = r[r < 0].std(ddof=1)
    sharpe = r.mean() / std * np.sqrt(252) if std > 0 else np.nan
    sortino = r.mean() / downside * np.sqrt(252) if pd.notna(downside) and downside > 0 else np.nan
    dd = curve / curve.cummax() - 1; max_dd = dd.min()
    cagr = curve.iloc[-1] ** (252 / len(test)) - 1
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    wins = r[r > 0]; losses = r[r < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) else np.inf

    print("\n" + "=" * 64)
    print("FROZEN OUT-OF-SAMPLE PERFORMANCE REPORT")
    print("=" * 64)
    print(f"Development: {development['Date'].iloc[0].date()} -> {development['Date'].iloc[-1].date()}")
    print(f"Test:        {test['Date'].iloc[0].date()} -> {test['Date'].iloc[-1].date()}")
    print(f"Frozen prediction cutoff: {pred_cutoff:.6f}")
    print(f"Frozen risk cutoff: {risk_cutoff:.6f}")
    print(f"Signals: {int(test['Signal'].sum())}")
    print(f"Strategy Return: {(curve.iloc[-1]-1):.4%}")
    print(f"Buy & Hold Return: {(bh_curve.iloc[-1]-1):.4%}")
    print(f"Outperformance: {(curve.iloc[-1]-bh_curve.iloc[-1]):.4%}")
    print(f"CAGR: {cagr:.4%}")
    print(f"Annualised Volatility: {std*np.sqrt(252):.4%}")
    print(f"Sharpe: {sharpe:.4f}")
    print(f"Sortino: {sortino:.4f}")
    print(f"Maximum Drawdown: {max_dd:.4%}")
    print(f"Calmar: {calmar:.4f}")
    print(f"Average Exposure: {test['Position'].mean():.2%}")
    print(f"Trades: {(test['Trade'] > 0).sum()}")
    print(f"Win Rate: {(r > 0).mean():.2%}")
    print(f"Profit Factor: {pf:.4f}")
    print(f"Average Win: {wins.mean() if len(wins) else 0:.6f}")
    print(f"Average Loss: {losses.mean() if len(losses) else 0:.6f}")
    print(f"Direction Accuracy: {(np.sign(test['Future_Return']) == np.sign(test['Predicted_Return'])).mean():.4f}")
    print("\nREGIME PERFORMANCE")
    for regime, g in test.groupby("Regime"):
        acc = (np.sign(g["Future_Return"]) == np.sign(g["Predicted_Return"])).mean()
        print(f"Regime {int(regime)}: samples={len(g)}, mean return={g['Future_Return'].mean():.4%}, direction accuracy={acc:.2%}, signals={int(g['Signal'].sum())}")


if __name__ == "__main__":
    main()
