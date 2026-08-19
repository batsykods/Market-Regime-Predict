import joblib
import pandas as pd


MODEL_PATH = "data/models/xgboost_signal_model.pkl"
INPUT_PATH = "data/processed/nifty50_signal_data.csv"


FEATURES = [
    "Log_Return",
    "Volatility_20",
    "SMA_20",
    "SMA_50",
    "Momentum_20",
    "Volume_Change",
    "Drawdown",
    "Regime"
]


LABELS = {
    0: "HOLD",
    1: "BUY",
    2: "SELL"
}


def main():

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(INPUT_PATH)

    latest = df.iloc[[-1]]

    X = latest[FEATURES]

    probabilities = model.predict_proba(X)[0]

    prediction = model.predict(X)[0]

    print("\nLatest Market Signal")
    print("=" * 40)

    print(
        f"Date: {latest['Date'].iloc[0]}"
    )

    print(
        f"Regime: {latest['Regime'].iloc[0]}"
    )

    print(
        f"Signal: {LABELS[prediction]}"
    )

    print("\nProbabilities:")

    print(
        f"HOLD: {probabilities[0]:.2%}"
    )

    print(
        f"BUY:  {probabilities[1]:.2%}"
    )

    print(
        f"SELL: {probabilities[2]:.2%}"
    )


if __name__ == "__main__":
    main()