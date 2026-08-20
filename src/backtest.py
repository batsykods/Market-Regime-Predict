import pandas as pd
import numpy as np

from src.performance import calculate_metrics


DATA_PATH = "data/processed/walk_forward_predictions.csv"

TRANSACTION_COST = 0.001
CONFIDENCE_THRESHOLD = 0.60


def main():

    df = pd.read_csv(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=[
            "Prediction",
            "Probability",
            "Future_Return"
        ]
    ).reset_index(drop=True)

    # -------------------------
    # Generate position
    # -------------------------

    df["Position"] = 0

    strong_buy = (
        (df["Prediction"] == 1) &
        (df["Probability"] >= CONFIDENCE_THRESHOLD)
    )

    strong_sell = (
        (df["Prediction"] == 2) &
        (df["Probability"] >= CONFIDENCE_THRESHOLD)
    )

    df.loc[strong_buy, "Position"] = 1
    df.loc[strong_sell, "Position"] = -1

    # -------------------------
    # Strategy return
    # -------------------------

    df["Strategy_Return"] = (
        df["Position"] *
        df["Future_Return"]
    )

    # -------------------------
    # Transaction costs
    # -------------------------

    df["Trade"] = (
        df["Position"]
        .diff()
        .abs()
        .fillna(0)
    )

    df["Transaction_Cost"] = (
        df["Trade"] *
        TRANSACTION_COST
    )

    df["Strategy_Return"] -= (
        df["Transaction_Cost"]
    )

    # -------------------------
    # Equity curves
    # -------------------------

    df["Strategy_Equity"] = (
        1 + df["Strategy_Return"]
    ).cumprod()

    df["BuyHold_Equity"] = (
        1 + df["Future_Return"]
    ).cumprod()

    # -------------------------
    # Metrics
    # -------------------------

    metrics = calculate_metrics(df)

    buy_hold_return = (
        df["BuyHold_Equity"].iloc[-1] - 1
    )

    print("\nFINAL BACKTEST")
    print("=" * 45)

    for name, value in metrics.items():

        if name == "Sharpe":
            print(f"{name}: {value:.2f}")

        else:
            print(f"{name}: {value:.2%}")

    print(
        f"Buy & Hold: {buy_hold_return:.2%}"
    )

    print(
        f"Trades: {int((df['Trade'] > 0).sum())}"
    )

    # Save results
    df.to_csv(
        "data/processed/backtest_results.csv",
        index=False
    )

    print(
        "\nSaved:"
        "\ndata/processed/backtest_results.csv"
    )


if __name__ == "__main__":
    main()

    from src.performance import calculate_metrics
    metrics = calculate_metrics(df)

    print("\nPERFORMANCE")
    print("=" * 40)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")