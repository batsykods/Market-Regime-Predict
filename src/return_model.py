import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


DATA_PATH = "data/processed/walk_forward_regimes.csv"
MODEL_PATH = "data/models/return_model.pkl"

FEATURES = [
     "Log_Return",
    "Volatility_20",
    "SMA_20",
    "SMA_50",
    "Momentum_20",
    "Volume_Change",
    "Drawdown",
    "Regime",
    "Return_Lag1",
    "Return_Lag2",
    "Return_Lag3",
    "Volatility_Lag1",
    "Momentum_Lag1",
]


def main():

    print("Loading data...")

    df = pd.read_csv(DATA_PATH)
    df["Return_Lag1"] = df["Return"].shift(1)
    df["Return_Lag2"] = df["Return"].shift(2)
    df["Return_Lag3"] = df["Return"].shift(3)
    df["Volatility_Lag1"] = df["Volatility_20"].shift(1)
    df["Momentum_Lag1"] = df["Momentum_20"].shift(1)

    # Next-day return
    df["Target_Return"] = df["Return"].shift(-1)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=FEATURES + ["Target_Return"]
    ).reset_index(drop=True)

    print(f"Training rows: {len(df)}")

    X = df[FEATURES].astype(float)
    y = df["Target_Return"].astype(float)

    # Time-based split
    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    print("Training model...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    mae = np.mean(
        np.abs(predictions - y_test)
    )

    rmse = np.sqrt(
        np.mean(
            (predictions - y_test) ** 2
        )
    )

    direction_accuracy = np.mean(
        np.sign(predictions)
        == np.sign(y_test)
    )

    print("\nRESULTS")
    print("=" * 40)
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(
        f"Direction Accuracy: "
        f"{direction_accuracy:.4f}"
    )

    os.makedirs(
        "data/models",
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nModel saved: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()