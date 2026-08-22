import numpy as np
import pandas as pd

INPUT = "data/processed/walk_forward_regime_ensemble.csv"


def evaluate(g):
    g = g.sort_values("Date").copy()
    pred = g["Predicted_Return"]
    actual = g["Future_Return"]
    eligible = pred > 0
    if not eligible.any():
        strategy = pd.Series(0.0, index=g.index)
    else:
        # Evaluation only: rank-based exposure, no threshold optimisation.
        rank = pred.rank(pct=True)
        position = np.where((pred > 0) & (rank >= 0.70), np.minimum(0.15, rank), 0.0)
        market = g["Close"].pct_change().fillna(0.0)
        strategy = pd.Series(position, index=g.index) * market
    equity = (1 + strategy).cumprod()
    bh = (1 + g["Close"].pct_change().fillna(0.0)).cumprod()
    return {
        "test_start": g["Date"].iloc[0].date(),
        "test_end": g["Date"].iloc[-1].date(),
        "samples": len(g),
        "signals": int(eligible.sum()),
        "strategy_return": equity.iloc[-1] - 1,
        "buy_hold_return": bh.iloc[-1] - 1,
        "outperformance": equity.iloc[-1] - bh.iloc[-1],
        "max_drawdown": (equity / equity.cummax() - 1).min(),
        "exposure": strategy.ne(0).mean(),
        "trades": int((pd.Series(np.where(strategy != 0, 1, 0), index=g.index).diff().abs().fillna(0) > 0).sum()),
    }


def main():
    df = pd.read_csv(INPUT)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    # Three non-overlapping chronological test windows.
    windows = np.array_split(df, 3)
    rows = [evaluate(w) for w in windows]
    result = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print("REGIME-ENSEMBLE NON-OVERLAPPING OOS VALIDATION")
    print("=" * 72)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSUMMARY")
    print(f"Windows beating Buy & Hold: {(result.outperformance > 0).sum()}/{len(result)}")
    print(f"Positive strategy windows: {(result.strategy_return > 0).sum()}/{len(result)}")
    print(f"Median outperformance: {result.outperformance.median():.4%}")


if __name__ == "__main__": main()
