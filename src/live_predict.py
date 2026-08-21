import joblib
import numpy as np
from src.live_data import get_live_data


MODEL_PATH = "data/models/xgboost_signal_model.pkl"
HMM_PATH = "data/models/regime_model.pkl"
SCALER_PATH = "data/models/regime_scaler.pkl"


FEATURES = [
    "Log_Return",
    "Volatility_20",
    "SMA_20",
    "SMA_50",
    "Momentum_20",
    "Volume_Change",
    "Drawdown",
]


def main():

    # 1. Load live market data
    df = get_live_data()

    if df.empty:
     raise ValueError(
        "Live market data is empty. "
        "Check Yahoo Finance connection."
    )

    latest = df.iloc[[-1]].copy()

    # 2. Load HMM and scaler
    hmm = joblib.load(HMM_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 3. HMM features
    regime_features = [
        "Log_Return",
        "Volatility_20",
        "Momentum_20"
    ]

    X_regime = latest[
        regime_features
    ].astype(float)

    X_scaled = scaler.transform(
        X_regime
    )

    # 4. Detect raw HMM state
    raw_state = int(
        hmm.predict(X_scaled)[0]
    )

    # IMPORTANT:
    # This mapping must match the mapping
    # used when training XGBoost.
    #
    # Change these only if your original
    # regime mapping is different.
    regime_mapping = {
        0: 0,
        1: 1,
        2: 2
    }

    regime = regime_mapping[raw_state]

    # 5. Load XGBoost
    xgb = joblib.load(
        MODEL_PATH
    )

    # 6. Add regime to features
    latest["Regime"] = regime

    xgb_features = [
        "Log_Return",
        "Volatility_20",
        "SMA_20",
        "SMA_50",
        "Momentum_20",
        "Volume_Change",
        "Drawdown",
        "Regime"
    ]

    X_live = latest[
        xgb_features
    ].astype(float)

    # 7. Predict signal
    prediction = int(
        xgb.predict(X_live)[0]
    )

    probabilities = (
        xgb.predict_proba(X_live)[0]
    )

    signal_map = {
        0: "HOLD",
        1: "BUY",
        2: "SELL"
    }

    # 8. Output
    print("\nLIVE MODEL PREDICTION")
    print("=" * 40)

    print(
        f"Date: {latest['Date'].iloc[0]}"
    )

    print(
        f"Close: {latest['Close'].iloc[0]:.2f}"
    )

    print(
        f"Raw HMM State: {raw_state}"
    )

    print(
        f"Regime: {regime}"
    )

    print(
        f"Signal: {signal_map[prediction]}"
    )

    print(
        f"HOLD Probability: "
        f"{probabilities[0]:.2%}"
    )

    print(
        f"BUY Probability: "
        f"{probabilities[1]:.2%}"
    )

    print(
        f"SELL Probability: "
        f"{probabilities[2]:.2%}"
    )


if __name__ == "__main__":
    main()