import os
import numpy as np
import pandas as pd


INPUT_PATH = "data/processed/walk_forward_results.csv"
OUTPUT_PATH = "data/processed/final_backtest.csv"

INITIAL_CAPITAL = 100000
TRANSACTION_COST = 0.001


def load_data():

    df = pd.read_csv(INPUT_PATH)

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    return df


def create_positions(df):

    # 1 = BUY
    # 2 = SELL
    # 0 = HOLD

    position = 0
    positions = []

    for prediction in df["Prediction"]:

        if prediction == 1:
            position = 1

        elif prediction == 2:
            position = 0

        positions.append(position)

    df["Position"] = positions

    # Signal is generated after today's close.
    # Execute from the next trading day.
    df["Position"] = (
        df["Position"]
        .shift(1)
        .fillna(0)
    )

    return df


def calculate_returns(df):

    # Actual next-day return is already
    # stored in the walk-forward dataset.
    df["Market_Return"] = (
        df["Future_Return"]
    )

    df["Strategy_Return"] = (
        df["Position"]
        * df["Market_Return"]
    )

    # Detect entry/exit
    df["Trade"] = (
        df["Position"]
        .diff()
        .abs()
        .fillna(
            df["Position"].abs()
        )
    )

    df["Transaction_Cost"] = (
        df["Trade"]
        * TRANSACTION_COST
    )

    df["Strategy_Return_After_Cost"] = (
        df["Strategy_Return"]
        - df["Transaction_Cost"]
    )

    return df


def calculate_equity(df):

    df["Strategy_Equity"] = (
        INITIAL_CAPITAL
        * (
            1
            + df[
                "Strategy_Return_After_Cost"
            ]
        ).cumprod()
    )

    df["Buy_Hold_Equity"] = (
        INITIAL_CAPITAL
        * (
            1
            + df["Market_Return"]
        ).cumprod()
    )

    return df


def calculate_metrics(df):

    strategy_returns = (
        df[
            "Strategy_Return_After_Cost"
        ]
        .dropna()
    )

    equity = df["Strategy_Equity"]

    total_return = (
        equity.iloc[-1]
        / equity.iloc[0]
        - 1
    )

    years = (
        len(strategy_returns)
        / 252
    )

    cagr = (
        (
            equity.iloc[-1]
            / equity.iloc[0]
        )
        ** (1 / years)
        - 1
    )

    volatility = (
        strategy_returns.std()
        * np.sqrt(252)
    )

    sharpe = (
        strategy_returns.mean()
        / strategy_returns.std()
        * np.sqrt(252)
    )

    rolling_max = equity.cummax()

    drawdown = (
        equity / rolling_max
        - 1
    )

    max_drawdown = drawdown.min()

    win_rate = (
        (strategy_returns > 0).sum()
        / len(strategy_returns)
    )

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualised Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": max_drawdown,
        "Win Rate": win_rate
    }


def main():

    print("Loading walk-forward predictions...")

    df = load_data()

    print(
        f"Rows: {len(df)}"
    )

    df = create_positions(df)

    df = calculate_returns(df)

    df = calculate_equity(df)

    metrics = calculate_metrics(df)

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("FINAL OUT-OF-SAMPLE BACKTEST")
    print("=" * 60)

    print(
        f"\nInitial Capital: "
        f"₹{INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Final Strategy Value: "
        f"₹{df['Strategy_Equity'].iloc[-1]:,.2f}"
    )

    print(
        f"Final Buy & Hold Value: "
        f"₹{df['Buy_Hold_Equity'].iloc[-1]:,.2f}"
    )

    print("\nPerformance:")

    for name, value in metrics.items():

        print(
            f"{name}: {value:.4f}"
        )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()