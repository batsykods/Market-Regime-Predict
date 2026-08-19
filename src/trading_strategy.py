import pandas as pd


BUY = 1
HOLD = 0
SELL = -1


def generate_position(
    prediction,
    probability,
    regime
):

    if probability < 0.60:
        return HOLD

    # Adjust these mappings after
    # analysing your actual HMM regimes.

    if regime == 1:  # Bull

        if prediction == 1:
            return BUY

        if prediction == 2:
            return HOLD

    elif regime == 2:  # Bear

        if prediction == 2:
            return SELL

        if prediction == 1:
            return HOLD

    elif regime == 0:  # Sideways

        if prediction == 1:
            return 0.25

        if prediction == 2:
            return -0.25

    return HOLD


def apply_strategy(df):

    df = df.copy()

    df["Position"] = df.apply(
        lambda row: generate_position(
            row["Prediction"],
            row["Probability"],
            row["Regime"]
        ),
        axis=1
    )

    return df