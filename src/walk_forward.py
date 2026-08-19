import pandas as pd
import numpy as np

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


FEATURES = [
    "Log_Return",
    "Volatility_20",
    "SMA_20",
    "SMA_50",
    "Momentum_20",
    "Volume_Change",
    "Drawdown",
]


def train_hmm(train):

    X = train[
        [
            "Log_Return",
            "Volatility_20",
            "Momentum_20"
        ]
    ]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    hmm = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )

    hmm.fit(X_scaled)

    train = train.copy()

    train["Regime"] = hmm.predict(
        X_scaled
    )

    return hmm, scaler, train


def add_regime(
    model,
    scaler,
    df
):

    X = df[
        [
            "Log_Return",
            "Volatility_20",
            "Momentum_20"
        ]
    ]

    X_scaled = scaler.transform(X)

    df = df.copy()

    df["Regime"] = model.predict(
        X_scaled
    )

    return df


def train_xgboost(train):

    features = FEATURES + ["Regime"]

    X = train[features]
    y = train["Target"]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(X, y)

    return model

def walk_forward(df):

    predictions = []

    initial_train = 1000

    test_size = 50

    for start in range(
        initial_train,
        len(df),
        test_size
    ):

        train = df.iloc[:start].copy()

        test = df.iloc[
            start:start + test_size
        ].copy()

        if len(test) == 0:
            break

        # HMM
        hmm, scaler, train = train_hmm(
            train
        )

        test = add_regime(
            hmm,
            scaler,
            test
        )

        # XGBoost
        model = train_xgboost(
            train
        )

        features = FEATURES + [
            "Regime"
        ]

        test["Prediction"] = model.predict(
            test[features]
        )

        test["Probability"] = (
            model.predict_proba(
                test[features]
            ).max(axis=1)
        )

        predictions.append(test)

    return pd.concat(
        predictions,
        ignore_index=True
    )