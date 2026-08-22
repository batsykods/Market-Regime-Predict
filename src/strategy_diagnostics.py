import numpy as np
import pandas as pd

INPUT_PATH = "data/processed/walk_forward_results.csv"
TRADING_DAYS = 252
HOLDING_DAYS = 5
TRANSACTION_COST = 0.001
INITIAL_CAPITAL = 100000


def position_size(row, prediction_cutoff, risk_cutoff):
    predicted = float(row["Predicted_Return"])
    volatility = max(float(row["Volatility_20"]), 1e-6)
    regime = int(row["Regime"])
    edge = predicted / (volatility * np.sqrt(HOLDING_DAYS))

    if predicted < prediction_cutoff or edge < risk_cutoff:
        return 0.0

    if edge < 0.50:
        position = 0.15
    elif edge < 0.75:
        position = 0.30
    elif edge < 1.00:
        position = 0.45
    else:
        position = 0.60

    if regime == 0:
        position *= 0.60
    elif regime == 2:
        position *= 0.85

    if volatility > 0.30:
        position *= 0.25
    elif volatility > 0.20:
        position *= 0.50
    elif volatility > 0.15:
        position *= 0.75

    return min(position, 0.60)


def build_strategy(df, prediction_quantile, risk_quantile):
    df = df.copy()
    risk_ratio = df["Predicted_Return"] / (df["Volatility_20"] * np.sqrt(HOLDING_DAYS))
    warmup = max(50, int(len(df) * 0.20))
    cut_pred = np.full(len(df), np.nan)
    cut_risk = np.full(len(df), np.nan)

    for i in range(warmup, len(df)):
        cut_pred[i] = df.loc[:i - 1, "Predicted_Return"].quantile(prediction_quantile)
        cut_risk[i] = risk_ratio.loc[:i - 1].quantile(risk_quantile)

    desired = np.zeros(len(df))
    for i in range(warmup, len(df)):
        desired[i] = position_size(df.iloc[i], cut_pred[i], cut_risk[i])

    position = np.zeros(len(df))
    active = 0.0
    remaining = 0
    for i in range(1, len(df)):
        signal = desired[i - 1]
        if signal > 0:
            active = max(active, signal)
            remaining = HOLDING_DAYS
        elif remaining > 0:
            remaining -= 1
        position[i] = active if remaining > 0 else 0.0
        if remaining == 0:
            active = 0.0

    market = df["Close"].pct_change().fillna(0.0)
    strategy = position * market
    trades = pd.Series(position).diff().abs().fillna(pd.Series(position).abs())
    net = strategy - trades * TRANSACTION_COST
    equity = INITIAL_CAPITAL * (1 + net).cumprod()

    return net, position, trades, equity


def metrics(returns, position, trades, equity):
    returns = pd.Series(returns)
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(returns) / TRADING_DAYS
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(TRADING_DAYS) if returns.std() else np.nan
    downside = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = returns.mean() * TRADING_DAYS / downside if downside else np.nan
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()
    return {
        "Total Return": total,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Calmar": cagr / abs(max_dd) if max_dd else np.nan,
        "Exposure": float(np.mean(position)),
        "Trades": int((trades > 0).sum()),
    }


def main():
    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    required = ["Close", "Future_Return", "Predicted_Return", "Regime", "Volatility_20"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).sort_values("Date").reset_index(drop=True)

    print("CONTROLLED A/B TEST — SAME DATASET")
    print(f"Rows: {len(df)}")
    print(f"Period: {df.Date.iloc[0].date()} -> {df.Date.iloc[-1].date()}")

    configs = {
        "Baseline 80/70": (0.80, 0.70),
        "Calibrated 65/60": (0.65, 0.60),
        "Intermediate 70/65": (0.70, 0.65),
    }

    for name, (pq, rq) in configs.items():
        net, position, trades, equity = build_strategy(df, pq, rq)
        m = metrics(net, position, trades, equity)
        print(f"\n{name}")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    benchmark = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
    print(f"\nBuy & Hold Return: {benchmark:.4f}")

    # Prediction quality by regime: diagnostics only, no parameter fitting.
    df["Correct_Direction"] = np.sign(df["Predicted_Return"]) == np.sign(df["Future_Return"])
    df["Abs_Error"] = (df["Predicted_Return"] - df["Future_Return"]).abs()
    regime_report = df.groupby("Regime").agg(
        Samples=("Future_Return", "size"),
        Mean_Future_Return=("Future_Return", "mean"),
        Direction_Accuracy=("Correct_Direction", "mean"),
        MAE=("Abs_Error", "mean"),
        Mean_Prediction=("Predicted_Return", "mean"),
    )
    print("\nMODEL DIAGNOSTICS BY REGIME")
    print(regime_report.to_string(float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
