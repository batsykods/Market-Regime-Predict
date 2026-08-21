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

def train_model(X, y):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X, y)
    return model


def main():
    print("Loading data...")

    df = pd.read_csv(DATA_PATH)

    df["Return_Lag1"] = df["Return"].shift(1)
    df["Return_Lag2"] = df["Return"].shift(2)
    df["Return_Lag3"] = df["Return"].shift(3)
    df["Volatility_Lag1"] = df["Volatility_20"].shift(1)
    df["Momentum_Lag1"] = df["Momentum_20"].shift(1)

    df["Target_Return"] = df["Return"].shift(-1)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=FEATURES + ["Target_Return"]
    ).reset_index(drop=True)

    initial_train = int(len(df) * 0.70)

    predictions = []
    actuals = []

    print(f"Rows: {len(df)}")
    print(f"Initial training size: {initial_train}")

    for i in range(initial_train, len(df)):
        train = df.iloc[:i]
        test = df.iloc[[i]]

        X_train = train[FEATURES].astype(float)
        y_train = train["Target_Return"].astype(float)

        X_test = test[FEATURES].astype(float)
        y_test = test["Target_Return"].iloc[0]

        model = train_model(
            X_train,
            y_train
        )

        prediction = model.predict(X_test)[0]

        predictions.append(prediction)
        actuals.append(y_test)

        if (i - initial_train + 1) % 25 == 0:
            print(
                f"Processed "
                f"{i - initial_train + 1}/"
                f"{len(df) - initial_train}"
            )

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae = np.mean(
        np.abs(predictions - actuals)
    )

    rmse = np.sqrt(
        np.mean(
            (predictions - actuals) ** 2
        )
    )

    direction_accuracy = np.mean(
        np.sign(predictions)
        == np.sign(actuals)
    )

    print("\nWALK-FORWARD RESULTS")
    print("=" * 40)
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(
        f"Direction Accuracy: "
        f"{direction_accuracy:.4f}"
    )

    results = df.iloc[
        initial_train:
    ][["Date", "Close"]].copy()

    results["Actual_Return"] = actuals
    results["Predicted_Return"] = predictions

    results.to_csv(
        "data/processed/"
        "walk_forward_returns.csv",
        index=False
    )

    print(
        "\nSaved: "
        "data/processed/"
        "walk_forward_returns.csv"
    )


if __name__ == "__main__":
    main()