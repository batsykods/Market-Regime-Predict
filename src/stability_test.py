import numpy as np
import pandas as pd

INPUT = "data/processed/walk_forward_results.csv"


def main():
    df = pd.read_csv(INPUT).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Predicted_Return", "Future_Return"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    chunks = np.array_split(df, 4)
    print("\n" + "=" * 72)
    print("TEMPORAL PREDICTION STABILITY")
    print("=" * 72)
    rows = []
    for i, g in enumerate(chunks, 1):
        rows.append({
            "Quarter": i,
            "Start": g["Date"].iloc[0].date(),
            "End": g["Date"].iloc[-1].date(),
            "Samples": len(g),
            "Mean_Prediction": g["Predicted_Return"].mean(),
            "Mean_Actual": g["Future_Return"].mean(),
            "MAE": (g["Future_Return"] - g["Predicted_Return"]).abs().mean(),
            "Direction_Accuracy": (np.sign(g["Future_Return"]) == np.sign(g["Predicted_Return"])).mean(),
            "Spearman": g["Predicted_Return"].corr(g["Future_Return"], method="spearman"),
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    corr = pd.Series([r["Spearman"] for r in rows])
    print(f"\nPositive-correlation periods: {(corr > 0).sum()}/{len(corr)}")
    print(f"Median period correlation: {corr.median():.4f}")
    print(f"Correlation range: {corr.min():.4f} to {corr.max():.4f}")


if __name__ == "__main__":
    main()
