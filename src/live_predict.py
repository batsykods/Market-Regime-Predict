import joblib
import pandas as pd

from src.live_data import get_live_data


MODEL_PATH = "data/models/xgboost_signal_model.pkl"
HMM_PATH = "data/models/regime_model.pkl"
SCALER_PATH = "data/models/regime_scaler.pkl"
MAPPING_PATH = "data/models/regime_mapping.pkl"

XGB_FEATURES = [
    "Log_Return",
    "Volatility_20",
    "SMA_20",
    "SMA_50",
    "Momentum_20",
    "Volume_Change",
    "Drawdown",
    "Regime",
]

REGIME_FEATURES = [
    "Log_Return",
    "Volatility_20",
    "Momentum_20",
]

SIGNAL_MAP = {
    0: "HOLD",
    1: "BUY",
    2: "SELL",
}


def main():
    df = get_live_data()

    if df.empty:
        raise ValueError(
            "Live market data is empty. Check Yahoo Finance connection."
        )

    latest = df.iloc[[-1]].copy()

    required = XGB_FEATURES[:-1] + REGIME_FEATURES
    missing = [col for col in set(required) if col not in latest.columns]
    if missing:
        raise ValueError(f"Missing live features: {missing}")

    hmm = joblib.load(HMM_PATH)
    scaler = joblib.load(SCALER_PATH)
    regime_mapping = joblib.load(MAPPING_PATH)
    xgb = joblib.load(MODEL_PATH)

    X_regime = latest[REGIME_FEATURES].astype(float)
    raw_state = int(hmm.predict(scaler.transform(X_regime))[0])

    if raw_state not in regime_mapping:
        raise ValueError(f"Unknown HMM state: {raw_state}")

    regime = int(regime_mapping[raw_state])
    latest["Regime"] = regime

    X_live = latest[XGB_FEATURES].astype(float)
    prediction = int(xgb.predict(X_live)[0])
    probabilities = xgb.predict_proba(X_live)[0]

    print("\nLIVE MODEL PREDICTION")
    print("=" * 40)
    print(f"Date: {latest['Date'].iloc[0]}")
    print(f"Close: {latest['Close'].iloc[0]:.2f}")
    print(f"Raw HMM State: {raw_state}")
    print(f"Regime: {regime}")
    print(f"Signal: {SIGNAL_MAP[prediction]}")
    print(f"HOLD Probability: {probabilities[0]:.2%}")
    print(f"BUY Probability: {probabilities[1]:.2%}")
    print(f"SELL Probability: {probabilities[2]:.2%}")


if __name__ == "__main__":
    main()
