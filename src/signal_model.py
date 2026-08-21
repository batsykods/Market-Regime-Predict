import os
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

INPUT_PATH = "data/processed/walk_forward_regimes.csv"
MODEL_PATH = "data/models/xgboost_signal_model.pkl"

FEATURES = [
    "Log_Return",
    "Volatility_20",
    "SMA_20",
    "SMA_50",
    "Momentum_20",
    "Volume_Change",
    "Drawdown",
    "Regime",
]

BUY_THRESHOLD = 0.005
SELL_THRESHOLD = -0.005


def load_data():
    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["Future_Return"] = df["Close"].shift(-1) / df["Close"] - 1
    df["Target"] = 0
    df.loc[df["Future_Return"] > BUY_THRESHOLD, "Target"] = 1
    df.loc[df["Future_Return"] < SELL_THRESHOLD, "Target"] = 2

    numeric = FEATURES + ["Target", "Future_Return"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=numeric).reset_index(drop=True)


def chronological_split(df):
    n = len(df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:validation_end], df.iloc[validation_end:]


def train_model(X, y):
    # Conservative model to reduce overfitting.
    model = XGBClassifier(
        n_estimators=150,
        max_depth=2,
        learning_rate=0.03,
        min_child_weight=8,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.2,
        reg_lambda=2.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X.astype(float), y.astype(int))
    return model


def evaluate(model, X, y, name):
    pred = model.predict(X.astype(float))
    print(f"\n{name} Accuracy: {accuracy_score(y, pred):.4f}")
    print(classification_report(y, pred, target_names=["HOLD", "BUY", "SELL"], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y, pred))


def main():
    df = load_data()
    train, validation, test = chronological_split(df)

    print("Dataset sizes:")
    print(f"Train:      {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test:       {len(test)}")

    model = train_model(train[FEATURES], train["Target"])

    evaluate(model, train[FEATURES], train["Target"], "TRAIN")
    evaluate(model, validation[FEATURES], validation["Target"], "VALIDATION")
    evaluate(model, test[FEATURES], test["Target"], "TEST")

    os.makedirs("data/models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
