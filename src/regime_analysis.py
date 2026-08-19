import pandas as pd
import numpy as np


INPUT_PATH = "data/processed/nifty50_regimes.csv"


def calculate_regime_statistics(df):

    statistics = df.groupby("Regime").agg(
        Mean_Return=("Return", "mean"),
        Median_Return=("Return", "median"),
        Return_Std=("Return", "std"),
        Mean_Momentum=("Momentum_20", "mean"),
        Mean_Volatility=("Volatility_20", "mean"),
        Positive_Days=("Return", lambda x: (x > 0).mean()),
        Observations=("Regime", "count")
    )

    statistics["Annualised_Return"] = (
        statistics["Mean_Return"] * 252
    )

    statistics["Annualised_Volatility"] = (
        statistics["Return_Std"] * np.sqrt(252)
    )

    return statistics


def calculate_regime_duration(df):

    regimes = df["Regime"].values

    durations = []

    current_regime = regimes[0]
    duration = 1

    for regime in regimes[1:]:

        if regime == current_regime:
            duration += 1

        else:
            durations.append(
                (current_regime, duration)
            )

            current_regime = regime
            duration = 1

    durations.append(
        (current_regime, duration)
    )

    duration_df = pd.DataFrame(
        durations,
        columns=["Regime", "Duration"]
    )

    return duration_df.groupby("Regime").agg(
        Average_Duration=("Duration", "mean"),
        Median_Duration=("Duration", "median"),
        Maximum_Duration=("Duration", "max"),
        Number_of_Periods=("Duration", "count")
    )


def main():

    df = pd.read_csv(INPUT_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    statistics = calculate_regime_statistics(df)

    duration_statistics = calculate_regime_duration(df)

    print("\nREGIME PERFORMANCE")
    print("=" * 60)
    print(statistics)

    print("\nREGIME DURATION")
    print("=" * 60)
    print(duration_statistics)


if __name__ == "__main__":
    main()