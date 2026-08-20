import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

INPUT_PATH = "data/processed/walk_forward_results.csv"


def main():

    df = pd.read_csv(INPUT_PATH)

    y_true = df["Actual"]
    y_pred = df["Prediction"]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print("\nWALK-FORWARD MODEL RESULTS")
    print("=" * 50)

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
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
            y_true,
            y_pred
        )
    )


if __name__ == "__main__":
    main()