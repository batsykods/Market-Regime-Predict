import os
import joblib
import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


INPUT_PATH = "data/processed/nifty50_features.csv"
OUTPUT_PATH = "data/processed/nifty50_regimes.csv"
MODEL_PATH = "data/models/regime_model.pkl"
SCALER_PATH = "data/models/regime_scaler.pkl"


FEATURES = [
    "Log_Return",
    "Volatility_20",
    "Momentum_20"
]


def load_data():
    df = pd.read_csv(INPUT_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    return df


def train_regime_model(df):

    X = df[FEATURES].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )

    model.fit(X_scaled)

    regimes = model.predict(X_scaled)

    df["Regime"] = regimes

    return df, model, scaler


def analyse_regimes(df):

    statistics = df.groupby("Regime").agg(
        Average_Return=("Return", "mean"),
        Volatility=("Return", "std"),
        Average_Momentum=("Momentum_20", "mean"),
        Average_Volatility=("Volatility_20", "mean"),
        Days=("Regime", "count")
    )

    return statistics


def main():

    os.makedirs("data/models", exist_ok=True)

    df = load_data()

    df, model, scaler = train_regime_model(df)

    statistics = analyse_regimes(df)

    print("\nRegime Statistics:")
    print(statistics)

    df.to_csv(OUTPUT_PATH, index=False)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(MODEL_PATH)
    print(SCALER_PATH)


if __name__ == "__main__":
    main()