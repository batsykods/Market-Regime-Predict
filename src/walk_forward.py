import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


INPUT_PATH = "data/processed/walk_forward_regimes.csv"
OUTPUT_PATH = "data/processed/walk_forward_results.csv"

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

TARGET = "Target"

BUY_THRESHOLD = 0.005
SELL_THRESHOLD = -0.005


def clean_data(df):

    # Convert features to numeric
    for column in FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    # Replace infinity values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove invalid rows
    df = df.dropna(
        subset=FEATURES + [TARGET]
    )

    df = df.reset_index(drop=True)

    df[TARGET] = df[TARGET].astype(int)

    return df


def create_target(df):

    # Tomorrow's return
    df["Future_Return"] = (
        df["Close"].shift(-1) / df["Close"]
    ) - 1

    # 0 = HOLD
    # 1 = BUY
    # 2 = SELL

    df["Target"] = 0

    df.loc[
        df["Future_Return"] > BUY_THRESHOLD,
        "Target"
    ] = 1

    df.loc[
        df["Future_Return"] < SELL_THRESHOLD,
        "Target"
    ] = 2

    # Last row has no future return
    df = df.dropna(
        subset=["Future_Return"]
    )

    return df


def train_model(X_train, y_train):

    X_train = X_train.astype(float)
    y_train = y_train.astype(int)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def main():

    print("Loading walk-forward regime data...")

    df = pd.read_csv(INPUT_PATH)

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    print(
        f"Raw rows: {len(df)}"
    )

    # Create BUY/HOLD/SELL target
    df = create_target(df)

    print(
        f"Rows after target creation: "
        f"{len(df)}"
    )

    # Clean features
    df = clean_data(df)

    print(
        f"Clean rows: {len(df)}"
    )

    # Initial historical training period
    initial_train_size = int(
        len(df) * 0.70
    )

    total_predictions = (
        len(df) - initial_train_size
    )

    print(
        f"Initial training size: "
        f"{initial_train_size}"
    )

    print(
        f"Walk-forward predictions: "
        f"{total_predictions}"
    )

    predictions = []

    for i in range(
        initial_train_size,
        len(df)
    ):

        # ONLY historical data
        train = df.iloc[:i]

        # Current unseen observation
        test = df.iloc[[i]]

        X_train = (
            train[FEATURES]
            .astype(float)
        )

        y_train = (
            train[TARGET]
            .astype(int)
        )

        X_test = (
            test[FEATURES]
            .astype(float)
        )

        # Safety checks
        if X_train.isnull().values.any():
            raise ValueError(
                "NaN detected in X_train."
            )

        if X_test.isnull().values.any():
            raise ValueError(
                "NaN detected in X_test."
            )

        # Train using historical data only
        model = train_model(
            X_train,
            y_train
        )

        # Predict unseen observation
        prediction = model.predict(
            X_test
        )[0]

        probabilities = model.predict_proba(
            X_test
        )[0]

        predictions.append({

            "Date":
                test["Date"].iloc[0],

            "Close":
                test["Close"].iloc[0],

            "Actual":
                test[TARGET].iloc[0],

            "Future_Return":
                test["Future_Return"].iloc[0],

            "Prediction":
                prediction,

            "HOLD_Probability":
                probabilities[0],

            "BUY_Probability":
                probabilities[1],

            "SELL_Probability":
                probabilities[2],

            "Regime":
                test["Regime"].iloc[0]
        })

        completed = (
            i - initial_train_size + 1
        )

        if completed % 25 == 0:

            print(
                f"Processed "
                f"{completed}/"
                f"{total_predictions}"
            )

    results = pd.DataFrame(
        predictions
    )

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print(
        "WALK-FORWARD XGBOOST COMPLETE"
    )
    print("=" * 60)

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print(
        f"Predictions generated: "
        f"{len(results)}"
    )

    print("\nPrediction distribution:")

    print(
        results["Prediction"]
        .value_counts()
        .sort_index()
    )

    print("\nActual distribution:")

    print(
        results["Actual"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()