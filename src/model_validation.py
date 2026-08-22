import numpy as np
import pandas as pd

INPUT = "data/processed/walk_forward_results.csv"


def main():
    df = pd.read_csv(INPUT).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Predicted_Return", "Future_Return", "Regime"])

    error = df["Future_Return"] - df["Predicted_Return"]
    df["Error"] = error
    df["Correct_Direction"] = np.sign(df["Future_Return"]) == np.sign(df["Predicted_Return"])

    print("\n" + "=" * 72)
    print("WALK-FORWARD MODEL VALIDATION")
    print("=" * 72)
    print(f"Samples: {len(df)}")
    print(f"MAE: {error.abs().mean():.6f}")
    print(f"RMSE: {np.sqrt((error ** 2).mean()):.6f}")
    print(f"Direction accuracy: {df['Correct_Direction'].mean():.4%}")
    print(f"Pearson correlation: {df['Predicted_Return'].corr(df['Future_Return']):.4f}")
    print(f"Spearman correlation: {df['Predicted_Return'].corr(df['Future_Return'], method='spearman'):.4f}")

    print("\nPREDICTION RANGE")
    print(df["Predicted_Return"].describe(percentiles=[.1,.25,.5,.75,.9]).to_string())

    print("\nREGIME VALIDATION")
    out = df.groupby("Regime").agg(
        Samples=("Future_Return", "size"),
        MAE=("Error", lambda x: x.abs().mean()),
        RMSE=("Error", lambda x: np.sqrt((x ** 2).mean())),
        Direction_Accuracy=("Correct_Direction", "mean"),
        Prediction_Correlation=("Predicted_Return", lambda x: x.corr(df.loc[x.index, "Future_Return"])),
        Mean_Actual_Return=("Future_Return", "mean"),
        Mean_Predicted_Return=("Predicted_Return", "mean"),
    )
    print(out.to_string(float_format=lambda x: f"{x:.6f}"))

    print("\nMODEL DECISION")
    corr = df["Predicted_Return"].corr(df["Future_Return"], method="spearman")
    if corr >= 0.20:
        print("KEEP: meaningful ranking signal; proceed to independent validation.")
    elif corr >= 0.10:
        print("WEAK: limited ranking signal; avoid further threshold optimisation.")
    else:
        print("REJECT: insufficient predictive ranking evidence.")


if __name__ == "__main__":
    main()
