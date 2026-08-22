import numpy as np
import pandas as pd

INPUT_PATH = "data/processed/walk_forward_results.csv"
HOLDING_DAYS = 5
PRED_Q = 0.70
RISK_Q = 0.65
COST = 0.001
REGIME_MULTIPLIERS = {0: 1.0, 1: 0.25, 2: 0.70}


def run_window(dev, test):
    pred_cut = dev["Predicted_Return"].quantile(PRED_Q)
    ratio_dev = dev["Predicted_Return"] / (dev["Volatility_20"] * np.sqrt(HOLDING_DAYS)).replace(0, np.nan)
    risk_cut = ratio_dev.quantile(RISK_Q)
    ratio = test["Predicted_Return"] / (test["Volatility_20"] * np.sqrt(HOLDING_DAYS)).replace(0, np.nan)
    signal = (test["Predicted_Return"] >= pred_cut) & (ratio >= risk_cut)
    pos = np.zeros(len(test))
    for i in range(len(test)):
        if signal.iloc[i]:
            e = ratio.iloc[i]
            p = 0.15 if e < .50 else .30 if e < .75 else .45 if e < 1 else .60
            p *= REGIME_MULTIPLIERS.get(int(test["Regime"].iloc[i]), .50)
            v = test["Volatility_20"].iloc[i]
            p *= .25 if v > .30 else .50 if v > .20 else .75 if v > .15 else 1
            pos[i] = min(p, .60)
    # Five-day persistent exposure, with signal generated at t and applied from t+1.
    held = np.zeros(len(test)); active = 0.; remaining = 0
    for i in range(1, len(test)):
        if pos[i-1] > 0:
            active = max(active, pos[i-1]); remaining = HOLDING_DAYS
        elif remaining > 0:
            remaining -= 1
        held[i] = active if remaining > 0 else 0
        if remaining == 0: active = 0
    ret = test["Close"].pct_change().fillna(0).to_numpy()
    strat = held * ret
    trade = np.abs(np.diff(np.r_[0, held]))
    net = strat - trade * COST
    curve = np.cumprod(1 + net)
    bh = np.cumprod(1 + ret)
    active = net[held > 0]
    wins = active[active > 0]; losses = active[active < 0]
    dd = curve / np.maximum.accumulate(curve) - 1
    return {
        "strategy_return": curve[-1]-1,
        "buy_hold_return": bh[-1]-1,
        "outperformance": curve[-1]-bh[-1],
        "max_drawdown": dd.min(),
        "exposure": held.mean(),
        "trades": int((trade > 0).sum()),
        "signals": int(signal.sum()),
        "profit_factor": wins.sum()/abs(losses.sum()) if len(losses) else np.inf,
        "win_rate": (active > 0).mean() if len(active) else 0,
        "prediction_cutoff": pred_cut,
        "risk_cutoff": risk_cut,
    }


def main():
    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    cols = ["Date", "Close", "Predicted_Return", "Volatility_20", "Regime"]
    df = df.sort_values("Date").dropna(subset=cols).reset_index(drop=True)
    n = len(df)
    # Rolling chronological validation: each test block is unseen when its thresholds are selected.
    windows = [(0.45, .15), (.55, .15), (.65, .15)]
    rows = []
    for d_frac, t_frac in windows:
        d_end = int(n*d_frac); t_end = min(n, d_end + int(n*t_frac))
        if t_end <= d_end: continue
        result = run_window(df.iloc[:d_end], df.iloc[d_end:t_end].reset_index(drop=True))
        result["development_end"] = df["Date"].iloc[d_end-1].date()
        result["test_start"] = df["Date"].iloc[d_end].date()
        result["test_end"] = df["Date"].iloc[t_end-1].date()
        rows.append(result)
    out = pd.DataFrame(rows)
    print("\n" + "="*72)
    print("ROLLING OUT-OF-SAMPLE ROBUSTNESS TEST")
    print("="*72)
    print(out[["development_end","test_start","test_end","signals","strategy_return","buy_hold_return","outperformance","max_drawdown","exposure","trades","profit_factor","win_rate"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPASS CRITERIA")
    print("Positive outperformance in each window:", bool((out.outperformance > 0).all()))
    print("Positive strategy return in each window:", bool((out.strategy_return > 0).all()))
    print("Median profit factor:", f"{out.profit_factor.replace([np.inf], np.nan).median():.4f}")

if __name__ == "__main__":
    main()
