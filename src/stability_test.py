import numpy as np
import pandas as pd

INPUT = "data/processed/walk_forward_results.csv"


def main():
    df = pd.read_csv(INPUT).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Predicted_Return", "Future_Return", "Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    n = len(df)
    boundaries = np.linspace(0, n, 5, dtype=int)

    print("\n" + "=" * 72)
    print("TEMPORAL PREDICTION STABILITY")
    print("=" * 72)

    rows = []
    for i in range(4):
        g = df.iloc[boundaries[i]:boundaries[i + 1]].copy()
        actual = g["Future_Return"]
        pred = g["Predicted_Return"]
        rows.append({
            "Quarter": i + 1,
            "Start": g["Date"].iloc[0].date(),
            "End": g["Date"].iloc[-1].date(),
            "Samples": len(g),
            "Mean_Prediction": pred.mean(),
            "Mean_Actual": actual.mean(),
            "MAE": (actual - pred).abs().mean(),
            "Direction_Accuracy": (np.sign(actual) == np.sign(pred)).mean(),
            "Spearman": pred.corr(actual, method="spearman"),
        })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    corr = result["Spearman"]
    print(f"\nPositive-correlation periods: {(corr > 0).sum()}/{len(corr)}")
    print(f"Median period correlation: {corr.median():.4f}")
    print(f"Correlation range: {corr.min():.4f} to {corr.max():.4f}")


if __name__ == "__main__":
    main()
