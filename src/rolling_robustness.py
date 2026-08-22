import numpy as np
import pandas as pd

INPUT_PATH = "data/processed/walk_forward_results.csv"
HOLDING_DAYS = 5
TRANSACTION_COST = 0.001
PREDICTION_QUANTILE = 0.70
RISK_QUANTILE = 0.65
REGIME_MULTIPLIERS = {0: 1.0, 1: 0.25, 2: 0.70}


def run_window(dev, test):
    pred_cutoff = dev["Predicted_Return"].quantile(PREDICTION_QUANTILE)
    dev_risk = dev["Predicted_Return"] / (dev["Volatility_20"] * np.sqrt(HOLDING_DAYS)).replace(0, np.nan)
    risk_cutoff = dev_risk.quantile(RISK_QUANTILE)

    test = test.copy().reset_index(drop=True)
    test["risk_ratio"] = test["Predicted_Return"] / (test["Volatility_20"] * np.sqrt(HOLDING_DAYS)).replace(0, np.nan)
    test["signal"] = (test["Predicted_Return"] >= pred_cutoff) & (test["risk_ratio"] >= risk_cutoff)

    def size(row):
        if not row["signal"]:
            return 0.0
        e = row["risk_ratio"]
        p = 0.15 if e < 0.50 else 0.30 if e < 0.75 else 0.45 if e < 1.00 else 0.60
        p *= REGIME_MULTIPLIERS.get(int(row["Regime"]), 0.50)
        v = row["Volatility_20"]
        p *= 0.25 if v > 0.30 else 0.50 if v > 0.20 else 0.75 if v > 0.15 else 1.0
        return min(p, 0.60)

    test["desired"] = test.apply(size, axis=1)
    pos = np.zeros(len(test)); active = 0.0; remaining = 0
    for i in range(1, len(test)):
        s = test.loc[i - 1, "desired"]
        if s > 0:
            active = max(active, s); remaining = HOLDING_DAYS
        elif remaining > 0:
            remaining -= 1
        pos[i] = active if remaining > 0 else 0.0
        if remaining == 0:
            active = 0.0

    test["position"] = pos
    test["market_return"] = test["Close"].pct_change().fillna(0.0)
    test["strategy_return"] = test["position"] * test["market_return"]
    test["trade"] = test["position"].diff().abs().fillna(test["position"].abs())
    test["net"] = test["strategy_return"] - test["trade"] * TRANSACTION_COST

    sr = (1 + test["net"]).prod() - 1
    bh = (1 + test["market_return"]).prod() - 1
    dd = ((1 + test["net"]).cumprod() / (1 + test["net"]).cumprod().cummax() - 1).min()
    wins = test.loc[(test["position"] > 0) & (test["net"] > 0), "net"]
    losses = test.loc[(test["position"] > 0) & (test["net"] < 0), "net"]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else (np.inf if len(wins) else np.nan)

    return {
        "test_start": test["Date"].iloc[0].date(),
        "test_end": test["Date"].iloc[-1].date(),
        "signals": int(test["signal"].sum()),
        "strategy_return": sr,
        "buy_hold_return": bh,
        "outperformance": sr - bh,
        "max_drawdown": dd,
        "exposure": test["position"].mean(),
        "trades": int((test["trade"] > 0).sum()),
        "profit_factor": pf,
    }


def main():
    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").dropna(
        subset=["Date", "Close", "Future_Return", "Predicted_Return", "Regime", "Volatility_20"]
    ).reset_index(drop=True)

    n = len(df)
    # Three non-overlapping evaluation windows. Each window is tested only
    # after its own historical development sample, avoiding test reuse.
    train_endpoints = [int(n * 0.55), int(n * 0.65), int(n * 0.75)]
    test_endpoints = [int(n * 0.70), int(n * 0.80), n]
    rows = []
    for dev_end, test_end in zip(train_endpoints, test_endpoints):
        test_start = dev_end
        if test_end <= test_start:
            continue
        rows.append(run_window(df.iloc[:dev_end], df.iloc[test_start:test_end]))

    out = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print("NON-OVERLAPPING ROLLING OUT-OF-SAMPLE ROBUSTNESS")
    print("=" * 72)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nROBUSTNESS SUMMARY")
    print(f"Windows beating Buy & Hold: {(out['outperformance'] > 0).sum()}/{len(out)}")
    print(f"Positive strategy windows: {(out['strategy_return'] > 0).sum()}/{len(out)}")
    print(f"Median outperformance: {out['outperformance'].median():.4%}")
    print(f"Median profit factor: {out['profit_factor'].median():.4f}")


if __name__ == "__main__":
    main()
