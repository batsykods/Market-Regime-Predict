import numpy as np
import pandas as pd

INPUT = "data/processed/walk_forward_results.csv"


def main():
    df = pd.read_csv(INPUT)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Predicted_Return", "Future_Return", "Regime"])
    df["Decile"] = pd.qcut(df["Predicted_Return"], 10, labels=False, duplicates="drop") + 1

    print("\n" + "=" * 78)
    print("WALK-FORWARD PREDICTION CALIBRATION")
    print("=" * 78)
    print("No trading threshold is optimised here. This tests whether prediction rank contains information.")

    summary = df.groupby("Decile").agg(
        Samples=("Future_Return", "size"),
        Mean_Prediction=("Predicted_Return", "mean"),
        Mean_Actual=("Future_Return", "mean"),
        Median_Actual=("Future_Return", "median"),
        Std_Actual=("Future_Return", "std"),
        Hit_Rate=("Future_Return", lambda x: (x > 0).mean()),
    )
    print("\nDECILE CALIBRATION")
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))

    rank_corr = df[["Predicted_Return", "Future_Return"]].corr(method="spearman").iloc[0, 1]
    pearson = df["Predicted_Return"].corr(df["Future_Return"])
    top = df[df["Decile"] == df["Decile"].max()]["Future_Return"]
    bottom = df[df["Decile"] == df["Decile"].min()]["Future_Return"]

    print("\nGLOBAL DIAGNOSTICS")
    print(f"Spearman rank correlation: {rank_corr:.4f}")
    print(f"Pearson correlation:       {pearson:.4f}")
    print(f"Top-decile mean return:    {top.mean():.4%}")
    print(f"Bottom-decile mean return: {bottom.mean():.4%}")
    print(f"Top-bottom spread:         {(top.mean() - bottom.mean()):.4%}")

    print("\nREGIME CALIBRATION")
    regime = df.groupby(["Regime", "Decile"]).agg(
        Samples=("Future_Return", "size"),
        Mean_Prediction=("Predicted_Return", "mean"),
        Mean_Actual=("Future_Return", "mean"),
        Hit_Rate=("Future_Return", lambda x: (x > 0).mean()),
    )
    print(regime.to_string(float_format=lambda x: f"{x:.6f}"))

    monotonic = summary["Mean_Actual"].corr(pd.Series(summary.index, index=summary.index), method="spearman")
    print(f"\nDecile/actual monotonicity: {monotonic:.4f}")
    if rank_corr > 0.10 and top.mean() > bottom.mean():
        print("ASSESSMENT: evidence of useful prediction ranking")
    else:
        print("ASSESSMENT: weak prediction ranking; trading-rule optimisation should stop")


if __name__ == "__main__":
    main()
