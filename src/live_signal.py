import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor

HMM_PATH = "data/models/regime_model.pkl"
SCALER_PATH = "data/models/regime_scaler.pkl"
MAPPING_PATH = "data/models/regime_mapping.pkl"
TRAIN_PATH = "data/processed/walk_forward_regimes.csv"
PREDICTIONS_PATH = "data/processed/walk_forward_results.csv"

FEATURES = [
    "Log_Return", "Volatility_20", "Volatility_5", "Volatility_Ratio",
    "SMA_20", "SMA_50", "Price_SMA20_Ratio", "Price_SMA50_Ratio",
    "Momentum_5", "Momentum_20", "Momentum_60", "Volume_Change",
    "Drawdown", "ATR_14", "Regime"
]
REGIME_FEATURES = ["Log_Return", "Volatility_20", "Momentum_20"]
HOLDING_DAYS = 5
PREDICTION_QUANTILE = 0.70
RISK_QUANTILE = 0.65
REGIME_MULTIPLIERS = {0: 1.00, 1: 0.25, 2: 0.70}


def add_features(df):
    df = df.copy()
    close = df["Close"].astype(float)
    log_return = df["Log_Return"].astype(float)
    df["Volatility_5"] = log_return.rolling(5).std() * np.sqrt(252)
    df["Volatility_Ratio"] = df["Volatility_5"] / df["Volatility_20"].replace(0, np.nan)
    df["Price_SMA20_Ratio"] = close / df["SMA_20"] - 1
    df["Price_SMA50_Ratio"] = close / df["SMA_50"] - 1
    df["Momentum_5"] = close.pct_change(5)
    df["Momentum_60"] = close.pct_change(60)
    return df


def train_model():
    df = pd.read_csv(TRAIN_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = add_features(df.sort_values("Date"))
    df["Future_Return"] = df["Close"].shift(-HOLDING_DAYS) / df["Close"] - 1
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["Future_Return"])
    model = XGBRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.025,
        min_child_weight=10, subsample=0.75, colsample_bytree=0.75,
        reg_alpha=0.25, reg_lambda=3.0, objective="reg:squarederror",
        eval_metric="rmse", random_state=42
    )
    model.fit(df[FEATURES].astype(float), df["Future_Return"].astype(float))
    return model


def live_frame():
    df = yf.download("^NSEI", period="1y", interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("No live market data returned by Yahoo Finance.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df["Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility_20"] = df["Log_Return"].rolling(20).std() * np.sqrt(252)
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["Momentum_20"] = df["Close"] / df["Close"].shift(20) - 1
    df["Momentum_5"] = df["Close"].pct_change(5)
    df["Momentum_60"] = df["Close"].pct_change(60)
    df["Volatility_5"] = df["Log_Return"].rolling(5).std() * np.sqrt(252)
    df["Volatility_Ratio"] = df["Volatility_5"] / df["Volatility_20"].replace(0, np.nan)
    df["Price_SMA20_Ratio"] = df["Close"] / df["SMA_20"] - 1
    df["Price_SMA50_Ratio"] = df["Close"] / df["SMA_50"] - 1
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Drawdown"] = df["Close"] / df["Close"].cummax() - 1
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def get_snapshot(capital=100000.0):
    market = live_frame()
    latest = market.iloc[[-1]].copy()
    hmm = joblib.load(HMM_PATH)
    scaler = joblib.load(SCALER_PATH)
    mapping = joblib.load(MAPPING_PATH)
    raw_state = int(hmm.predict(scaler.transform(latest[REGIME_FEATURES].astype(float)))[0])
    regime = int(mapping[raw_state])
    latest["Regime"] = regime

    model = train_model()
    prediction = float(model.predict(latest[FEATURES].astype(float))[0])

    history = pd.read_csv(PREDICTIONS_PATH)
    history = history.replace([np.inf, -np.inf], np.nan).dropna(subset=["Predicted_Return", "Volatility_20"])
    pred_cutoff = history["Predicted_Return"].quantile(PREDICTION_QUANTILE)
    risk_ratio_history = history["Predicted_Return"] / (history["Volatility_20"] * np.sqrt(HOLDING_DAYS)).replace(0, np.nan)
    risk_cutoff = risk_ratio_history.quantile(RISK_QUANTILE)

    volatility = float(latest["Volatility_20"].iloc[0])
    risk = max(volatility * np.sqrt(HOLDING_DAYS), 1e-6)
    edge = prediction / risk
    eligible = prediction >= pred_cutoff and edge >= risk_cutoff

    if edge < 0.50: base = 0.15
    elif edge < 0.75: base = 0.30
    elif edge < 1.00: base = 0.45
    else: base = 0.60
    if volatility > 0.30: base *= 0.25
    elif volatility > 0.20: base *= 0.50
    elif volatility > 0.15: base *= 0.75
    position = min(base * REGIME_MULTIPLIERS.get(regime, 0.50), 0.60) if eligible else 0.0

    price = float(latest["Close"].iloc[0])
    atr = float(latest["ATR_14"].iloc[0])
    target = price * (1 + prediction)
    stop = max(price - 1.5 * atr, 0.0)
    expected_profit = capital * position * prediction
    signal = "BUY" if eligible and prediction > 0 else "HOLD"
    if prediction < 0 and eligible: signal = "SELL"

    return {
        "timestamp": str(latest["Date"].iloc[0]), "price": price,
        "regime": regime, "raw_state": raw_state, "signal": signal,
        "predicted_return": prediction, "prediction_percentile": float((history["Predicted_Return"] <= prediction).mean()),
        "risk_ratio": edge, "prediction_cutoff": float(pred_cutoff), "risk_cutoff": float(risk_cutoff),
        "position_size": position, "target_price": target, "stop_price": stop,
        "expected_profit": expected_profit, "holding_days": HOLDING_DAYS,
        "volatility_20": volatility, "atr_14": atr, "eligible": bool(eligible),
    }


def main():
    s = get_snapshot()
    print("\nLIVE 5-DAY TRADING SIGNAL")
    print("=" * 55)
    for k, v in s.items(): print(f"{k}: {v}")


if __name__ == "__main__":
    main()
