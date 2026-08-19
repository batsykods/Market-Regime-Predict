import os
import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


INPUT_PATH = "data/processed/nifty50_signal_data.csv"
MODEL_PATH = "data/models/xgboost_signal_model.pkl"


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


def load_data():
    df = pd.read_csv(INPUT_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.replace([float("inf"), float("-inf")], float("nan"))

    numeric_columns = [
        "Log_Return",
        "Volatility_20",
        "SMA_20",
        "SMA_50",
        "Momentum_20",
        "Volume_Change",
        "Drawdown",
        "Regime",
        "Target"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=numeric_columns
    )

    return df

def chronological_split(df):

    total = len(df)

    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)

    train = df.iloc[:train_end]
    validation = df.iloc[train_end:validation_end]
    test = df.iloc[validation_end:]

    return train, validation, test


def train_model(X_train, y_train):

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


def evaluate(model, X, y, dataset_name):

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    print(f"\n{dataset_name} Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "HOLD",
                "BUY",
                "SELL"
            ],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y,
            predictions
        )
    )


def main():

    df = load_data()

    train, validation, test = chronological_split(df)

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print("Dataset sizes:")
    print(f"Train:      {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test:       {len(test)}")

    model = train_model(
        X_train,
        y_train
    )

    evaluate(
        model,
        X_train,
        y_train,
        "TRAIN"
    )

    evaluate(
        model,
        X_validation,
        y_validation,
        "VALIDATION"
    )

    evaluate(
        model,
        X_test,
        y_test,
        "TEST"
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
        f"\nModel saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
