import numpy as np


MAX_POSITION = 1.0
MIN_POSITION = 0.0

TARGET_VOLATILITY = 0.15


def volatility_position_size(volatility):

    if volatility <= 0:
        return MAX_POSITION

    position = (
        TARGET_VOLATILITY /
        volatility
    )

    return np.clip(
        position,
        MIN_POSITION,
        MAX_POSITION
    )


def apply_risk_management(df):

    df = df.copy()

    df["Position_Size"] = df[
        "Volatility_20"
    ].apply(
        volatility_position_size
    )

    df["Final_Position"] = (
        df["Position"] *
        df["Position_Size"]
    )

    return df