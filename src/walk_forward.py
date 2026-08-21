import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix

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


def create_target(df):
    df["Future_Return"] = df["Close"].shift(-1) / df["Close"] - 1
    df["Target"] = 0
    df.loc[df["Future_Return"] > BUY_THRESHOLD, "Target"] = 1
    df.loc[df["Future_Return"] < SELL_THRESHOLD, "Target"] = 2
    return df.dropna(subset=["Future_Return"]).copy()


def clean_data(df):
    for column in FEATURES + [TARGET, "Future_Return"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + [TARGET, "Future_Return"])
    df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True)


def train_model(X_train, y_train):
    counts = y_train.value_counts().sort_index()
    total = len(y_train)
    class_weights = {
        int(cls): total / (len(counts) * count)
        for cls, count in counts.items()
    }
    sample_weight = y_train.map(class_weights).to_numpy()

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
    model.fit(X_train.astype(float), y_train.astype(int), sample_weight=sample_weight)
    return model


def main():
    print("Loading walk-forward regime data...")
    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    print(f"Raw rows: {len(df)}")
    df = create_target(df)
    print(f"Rows after target creation: {len(df)}")
    df = clean_data(df)
    print(f"Clean rows: {len(df)}")

    initial_train_size = int(len(df) * 0.70)
    total_predictions = len(df) - initial_train_size
    print(f"Initial training size: {initial_train_size}")
    print(f"Walk-forward predictions: {total_predictions}")

    predictions = []

    for i in range(initial_train_size, len(df)):
        train = df.iloc[:i]
        test = df.iloc[[i]]

        model = train_model(train[FEATURES], train[TARGET])
        X_test = test[FEATURES].astype(float)
        prediction = int(model.predict(X_test)[0])
        probabilities = model.predict_proba(X_test)[0]

        predictions.append({
            "Date": test["Date"].iloc[0],
            "Close": test["Close"].iloc[0],
            "Actual": int(test[TARGET].iloc[0]),
            "Future_Return": test["Future_Return"].iloc[0],
            "Prediction": prediction,
            "HOLD_Probability": probabilities[0],
            "BUY_Probability": probabilities[1],
            "SELL_Probability": probabilities[2],
            "Regime": int(test["Regime"].iloc[0]),
        })

        completed = i - initial_train_size + 1
        if completed % 25 == 0:
            print(f"Processed {completed}/{total_predictions}")

    results = pd.DataFrame(predictions)
    os.makedirs("data/processed", exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    y_true = results["Actual"].astype(int)
    y_pred = results["Prediction"].astype(int)

    print("\n" + "=" * 60)
    print("WALK-FORWARD XGBOOST COMPLETE")
    print("=" * 60)
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Predictions generated: {len(results)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["HOLD", "BUY", "SELL"], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nPrediction distribution:")
    print(results["Prediction"].value_counts().sort_index())
    print("\nActual distribution:")
    print(results["Actual"].value_counts().sort_index())


if __name__ == "__main__":
    main()
