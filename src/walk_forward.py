import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT_PATH = "data/processed/walk_forward_regimes.csv"
OUTPUT_PATH = "data/processed/walk_forward_results.csv"

FEATURES = [
    "Log_Return", "Volatility_20", "SMA_20", "SMA_50",
    "Momentum_20", "Volume_Change", "Drawdown", "Regime"
]

TARGET_HORIZON = 1
INITIAL_TRAIN_FRACTION = 0.70


def make_target(df):
    df = df.copy()
    df["Future_Return"] = df["Close"].shift(-TARGET_HORIZON) / df["Close"] - 1.0
    return df.dropna(subset=["Future_Return"]).copy()


def clean(df):
    cols = FEATURES + ["Future_Return"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols).reset_index(drop=True)


def train_model(X, y):
    model = XGBRegressor(
        n_estimators=150,
        max_depth=2,
        learning_rate=0.03,
        min_child_weight=8,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.2,
        reg_lambda=2.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
    )
    model.fit(X.astype(float), y.astype(float))
    return model


def main():
    print("Loading walk-forward regime data...")
    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = clean(make_target(df.sort_values("Date").reset_index(drop=True)))

    initial = int(len(df) * INITIAL_TRAIN_FRACTION)
    predictions = []
    print(f"Rows: {len(df)}")
    print(f"Initial training size: {initial}")
    print(f"Walk-forward predictions: {len(df) - initial}")

    for i in range(initial, len(df)):
        train = df.iloc[:i]
        test = df.iloc[[i]]
        model = train_model(train[FEATURES], train["Future_Return"])
        pred = float(model.predict(test[FEATURES].astype(float))[0])
        predictions.append({
            "Date": test["Date"].iloc[0],
            "Close": test["Close"].iloc[0],
            "Future_Return": test["Future_Return"].iloc[0],
            "Predicted_Return": pred,
            "Regime": int(test["Regime"].iloc[0]),
            "Volatility_20": float(test["Volatility_20"].iloc[0]),
        })
        n = i - initial + 1
        if n % 25 == 0:
            print(f"Processed {n}/{len(df) - initial}")

    results = pd.DataFrame(predictions)
    os.makedirs("data/processed", exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    y = results["Future_Return"]
    p = results["Predicted_Return"]
    direction = np.mean(np.sign(y) == np.sign(p))
    mae = mean_absolute_error(y, p)
    rmse = np.sqrt(mean_squared_error(y, p))
    corr = y.corr(p)

    print("\n" + "=" * 60)
    print("WALK-FORWARD RETURN MODEL COMPLETE")
    print("=" * 60)
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Predictions generated: {len(results)}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Direction Accuracy: {direction:.4f}")
    print(f"Prediction Correlation: {corr:.4f}")


if __name__ == "__main__":
    main()
