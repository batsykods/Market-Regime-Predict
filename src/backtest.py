import joblib
import pandas as pd
import numpy as np


DATA_PATH = "data/processed/nifty50_signal_data.csv"
MODEL_PATH = "data/models/xgboost_signal_model.pkl"

FEATURES = [
    "Log_Return",
    "Volatility_20",
    "SMA_20",
    "SMA_50",
    "Momentum_20",
    "Volume_Change",
    "Drawdown",
    "Regime"
]

TRANSACTION_COST = 0.001


def main():

    df = pd.read_csv(DATA_PATH)

    model = joblib.load(MODEL_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=FEATURES
    ).reset_index(drop=True)

    # Model prediction
    df["Prediction"] = model.predict(
        df[FEATURES]
    )

    # Convert prediction into position
    # BUY  = +1
    # HOLD =  0
    # SELL = -1

    df["Position"] = 0

    df.loc[
        df["Prediction"] == 1,
        "Position"
    ] = 1

    df.loc[
        df["Prediction"] == 2,
        "Position"
    ] = -1

    # Tomorrow's return
    df["Future_Return"] = (
        df["Close"].shift(-1) /
        df["Close"]
        - 1
    )

    # Strategy return
    df["Strategy_Return"] = (
        df["Position"] *
        df["Future_Return"]
    )

    # Position changes
    df["Trade"] = (
        df["Position"]
        .diff()
        .abs()
    )

    # Transaction costs
    df["Strategy_Return"] -= (
        df["Trade"] *
        TRANSACTION_COST
    )

    df = df.dropna(
        subset=["Strategy_Return"]
    )

    # Equity curves
    df["Strategy_Equity"] = (
        1 + df["Strategy_Return"]
    ).cumprod()

    df["BuyHold_Equity"] = (
        1 + df["Future_Return"]
    ).cumprod()

    # Performance
    strategy_return = (
        df["Strategy_Equity"].iloc[-1] - 1
    )

    buyhold_return = (
        df["BuyHold_Equity"].iloc[-1] - 1
    )

    volatility = (
        df["Strategy_Return"].std()
        * np.sqrt(252)
    )

    sharpe = (
        df["Strategy_Return"].mean()
        / df["Strategy_Return"].std()
        * np.sqrt(252)
    )

    cumulative = df["Strategy_Equity"]

    drawdown = (
        cumulative /
        cumulative.cummax()
        - 1
    )

    max_drawdown = drawdown.min()

    print("\nBACKTEST RESULTS")
    print("=" * 40)

    print(
        f"Strategy Return: "
        f"{strategy_return:.2%}"
    )

    print(
        f"Buy & Hold Return: "
        f"{buyhold_return:.2%}"
    )

    print(
        f"Annualised Volatility: "
        f"{volatility:.2%}"
    )

    print(
        f"Sharpe Ratio: "
        f"{sharpe:.2f}"
    )

    print(
        f"Maximum Drawdown: "
        f"{max_drawdown:.2%}"
    )

    print(
        f"Number of Trades: "
        f"{int((df['Trade'] > 0).sum())}"
    )


if __name__ == "__main__":
    main()

    from src.performance import calculate_metrics
    metrics = calculate_metrics(df)

    print("\nPERFORMANCE")
    print("=" * 40)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")