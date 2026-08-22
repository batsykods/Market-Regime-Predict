import numpy as np
import pandas as pd

INPUT = "data/processed/walk_forward_regime_ensemble.csv"


def evaluate(g):
    g = g.sort_values("Date").copy()
    pred = g["Predicted_Return"]
    eligible = pred > 0
    rank = pred.rank(pct=True)
    position = np.where((pred > 0) & (rank >= 0.70), np.minimum(0.15, rank), 0.0)
    market = g["Close"].pct_change().fillna(0.0)
    strategy = pd.Series(position, index=g.index) * market
    equity = (1 + strategy).cumprod()
    bh = (1 + market).cumprod()
    return {
        "test_start": g["Date"].iloc[0].date(),
        "test_end": g["Date"].iloc[-1].date(),
        "samples": len(g),
        "signals": int(eligible.sum()),
        "strategy_return": equity.iloc[-1] - 1,
        "buy_hold_return": bh.iloc[-1] - 1,
        "outperformance": equity.iloc[-1] - bh.iloc[-1],
        "max_drawdown": (equity / equity.cummax() - 1).min(),
        "exposure": float(np.mean(position > 0)),
        "trades": int(np.count_nonzero(np.diff((position > 0).astype(int)))),
    }


def main():
    df = pd.read_csv(INPUT)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    edges = np.linspace(0, n, 4, dtype=int)
    windows = [df.iloc[edges[i]:edges[i + 1]].copy() for i in range(3)]
    result = pd.DataFrame([evaluate(w) for w in windows])
    print("\n" + "=" * 72)
    print("REGIME-ENSEMBLE NON-OVERLAPPING OOS VALIDATION")
    print("=" * 72)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSUMMARY")
    print(f"Windows beating Buy & Hold: {(result.outperformance > 0).sum()}/{len(result)}")
    print(f"Positive strategy windows: {(result.strategy_return > 0).sum()}/{len(result)}")
    print(f"Median outperformance: {result.outperformance.median():.4%}")


if __name__ == "__main__":
    main()
