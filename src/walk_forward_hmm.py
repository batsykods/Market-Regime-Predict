import os
import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


INPUT_PATH = "data/processed/nifty50_features.csv"
OUTPUT_PATH = "data/processed/walk_forward_regimes.csv"


REGIME_FEATURES = [
    "Log_Return",
    "Volatility_20",
    "Momentum_20"
]

INITIAL_TRAIN_SIZE = 1000
RETRAIN_EVERY = 20
N_REGIMES = 3


def clean_data(df):

    for column in REGIME_FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=REGIME_FEATURES
    )

    df = df.reset_index(drop=True)

    return df


def train_hmm(train_data):

    X_train = train_data[
        REGIME_FEATURES
    ].astype(float)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_train
    )

    model = GaussianHMM(
        n_components=N_REGIMES,
        covariance_type="full",
        n_iter=500,
        random_state=42
    )

    model.fit(X_scaled)

    return model, scaler


def label_regimes(model, scaler, train_data):

    X_train = train_data[
        REGIME_FEATURES
    ].astype(float)

    X_scaled = scaler.transform(
        X_train
    )

    states = model.predict(
        X_scaled
    )

    temp = train_data.copy()

    temp["State"] = states

    statistics = temp.groupby(
        "State"
    ).agg(
        Mean_Return=("Return", "mean"),
        Mean_Volatility=(
            "Volatility_20",
            "mean"
        ),
        Mean_Momentum=(
            "Momentum_20",
            "mean"
        )
    )

    # Highest return = Bull
    bull_state = statistics[
        "Mean_Return"
    ].idxmax()

    # Lowest return = Bear
    bear_state = statistics[
        "Mean_Return"
    ].idxmin()

    remaining_states = [
        state
        for state in statistics.index
        if state not in [
            bull_state,
            bear_state
        ]
    ]

    regime_mapping = {
        bull_state: 1,
        bear_state: 2
    }

    if remaining_states:
        regime_mapping[
            remaining_states[0]
        ] = 0

    return regime_mapping, statistics


def main():

    print("Loading data...")

    df = pd.read_csv(
        INPUT_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    print(
        f"Raw rows: {len(df)}"
    )

    df = clean_data(df)

    print(
        f"Clean rows: {len(df)}"
    )

    if len(df) <= INITIAL_TRAIN_SIZE:
        raise ValueError(
            "Not enough data for walk-forward HMM."
        )

    regimes = []

    total_predictions = (
        len(df) - INITIAL_TRAIN_SIZE
    )

    for start in range(
        INITIAL_TRAIN_SIZE,
        len(df),
        RETRAIN_EVERY
    ):

        train_data = df.iloc[:start]

        end = min(
            start + RETRAIN_EVERY,
            len(df)
        )

        test_data = df.iloc[
            start:end
        ]

        print(
            f"\nTraining HMM:"
            f" {train_data['Date'].iloc[0].date()}"
            f" → "
            f"{train_data['Date'].iloc[-1].date()}"
        )

        print(
            f"Predicting:"
            f" {test_data['Date'].iloc[0].date()}"
            f" → "
            f"{test_data['Date'].iloc[-1].date()}"
        )

        # Train only on historical data
        model, scaler = train_hmm(
            train_data
        )

        # Determine meaning of states
        regime_mapping, statistics = (
            label_regimes(
                model,
                scaler,
                train_data
            )
        )

        X_test = test_data[
            REGIME_FEATURES
        ].astype(float)

        X_test_scaled = scaler.transform(
            X_test
        )

        states = model.predict(
            X_test_scaled
        )

        for date, state in zip(
            test_data["Date"],
            states
        ):

            regime = regime_mapping.get(
                state,
                0
            )

            regimes.append({
                "Date": date,
                "Regime": regime,
                "Raw_State": state
            })

    regime_df = pd.DataFrame(
        regimes
    )

    output = df.merge(
        regime_df,
        on="Date",
        how="inner"
    )

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print(
        "WALK-FORWARD HMM COMPLETE"
    )
    print("=" * 60)

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print(
        f"Regime observations: "
        f"{len(output)}"
    )

    print("\nRegime distribution:")

    print(
        output["Regime"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()