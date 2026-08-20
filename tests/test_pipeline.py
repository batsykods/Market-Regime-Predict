import os
import pandas as pd


def test_features_file_exists():

    path = "data/processed/nifty50_features.csv"

    assert os.path.exists(path)


def test_regime_file_exists():

    path = "data/processed/walk_forward_regimes.csv"

    assert os.path.exists(path)


def test_predictions_file_exists():

    path = "data/processed/walk_forward_results.csv"

    assert os.path.exists(path)


def test_backtest_file_exists():

    path = "data/processed/risk_adjusted_backtest.csv"

    assert os.path.exists(path)


def test_predictions_have_required_columns():

    path = "data/processed/walk_forward_results.csv"

    df = pd.read_csv(path)

    required = [
        "Date",
        "Actual",
        "Prediction",
        "BUY_Probability",
        "HOLD_Probability",
        "SELL_Probability"
    ]

    for column in required:

        assert column in df.columns


def test_predictions_are_valid():

    df = pd.read_csv(
        "data/processed/walk_forward_results.csv"
    )

    assert df["Prediction"].isin(
        [0, 1, 2]
    ).all()


def test_probabilities_are_valid():

    df = pd.read_csv(
        "data/processed/walk_forward_results.csv"
    )

    probabilities = (
        df["BUY_Probability"]
        + df["HOLD_Probability"]
        + df["SELL_Probability"]
    )

    assert ((probabilities - 1).abs() < 0.001).all()


def test_backtest_has_equity():

    df = pd.read_csv(
        "data/processed/risk_adjusted_backtest.csv"
    )

    assert "Strategy_Equity" in df.columns

    assert "Buy_Hold_Equity" in df.columns

    assert df["Strategy_Equity"].notna().all()


def test_no_negative_equity():

    df = pd.read_csv(
        "data/processed/risk_adjusted_backtest.csv"
    )

    assert (
        df["Strategy_Equity"] >= 0
    ).all()