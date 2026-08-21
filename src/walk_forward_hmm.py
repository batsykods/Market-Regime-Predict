import os
import joblib
import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

INPUT_PATH = "data/processed/nifty50_features.csv"
OUTPUT_PATH = "data/processed/walk_forward_regimes.csv"
MODEL_PATH = "data/models/regime_model.pkl"
SCALER_PATH = "data/models/regime_scaler.pkl"
MAPPING_PATH = "data/models/regime_mapping.pkl"
REGIME_FEATURES = ["Log_Return", "Volatility_20", "Momentum_20"]
INITIAL_TRAIN_SIZE = 1000
RETRAIN_EVERY = 20
N_REGIMES = 3


def clean_data(df):
    for column in REGIME_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=REGIME_FEATURES).reset_index(drop=True)


def train_hmm(train_data):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_data[REGIME_FEATURES].astype(float))
    model = GaussianHMM(n_components=N_REGIMES, covariance_type="full", n_iter=500, random_state=42)
    model.fit(X_scaled)
    return model, scaler


def build_mapping(model, scaler, train_data):
    states = model.predict(scaler.transform(train_data[REGIME_FEATURES].astype(float)))
    temp = train_data.copy()
    temp["State"] = states
    statistics = temp.groupby("State").agg(
        Mean_Return=("Return", "mean"),
        Mean_Volatility=("Volatility_20", "mean"),
    )
    bull_state = int(statistics["Mean_Return"].idxmax())
    bear_state = int(statistics["Mean_Return"].idxmin())
    remaining = [int(s) for s in statistics.index if int(s) not in {bull_state, bear_state}]
    mapping = {bull_state: 1, bear_state: 2}
    if remaining:
        mapping[remaining[0]] = 0
    return mapping


def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = clean_data(df.sort_values("Date").reset_index(drop=True))

    if len(df) <= INITIAL_TRAIN_SIZE:
        raise ValueError("Not enough data for walk-forward HMM.")

    regimes = []
    for start in range(INITIAL_TRAIN_SIZE, len(df), RETRAIN_EVERY):
        train_data = df.iloc[:start]
        test_data = df.iloc[start:min(start + RETRAIN_EVERY, len(df))]
        model, scaler = train_hmm(train_data)
        mapping = build_mapping(model, scaler, train_data)
        states = model.predict(scaler.transform(test_data[REGIME_FEATURES].astype(float)))
        for date, state in zip(test_data["Date"], states):
            regimes.append({"Date": date, "Regime": mapping[int(state)], "Raw_State": int(state)})

    output = df.merge(pd.DataFrame(regimes), on="Date", how="inner")
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    final_model, final_scaler = train_hmm(df)
    final_mapping = build_mapping(final_model, final_scaler, df)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(final_scaler, SCALER_PATH)
    joblib.dump(final_mapping, MAPPING_PATH)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved: {MODEL_PATH}")
    print(f"Saved: {SCALER_PATH}")
    print(f"Saved: {MAPPING_PATH}")


if __name__ == "__main__":
    main()
