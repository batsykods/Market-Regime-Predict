import numpy as np
import pandas as pd


INPUT_PATH = "data/processed/risk_adjusted_backtest.csv"

TRADING_DAYS = 252


def calculate_metrics(df):

    returns = df[
        "Strategy_Return_After_Cost"
    ].dropna()

    equity = df["Strategy_Equity"]

    # Total Return
    total_return = (
        equity.iloc[-1] / equity.iloc[0]
    ) - 1

    # CAGR
    years = len(returns) / TRADING_DAYS

    cagr = (
        (equity.iloc[-1] / equity.iloc[0])
        ** (1 / years)
    ) - 1

    # Annualised Volatility
    volatility = (
        returns.std()
        * np.sqrt(TRADING_DAYS)
    )

    # Sharpe
    sharpe = (
        returns.mean()
        / returns.std()
        * np.sqrt(TRADING_DAYS)
    )

    # Downside deviation
    negative_returns = returns[
        returns < 0
    ]

    downside_deviation = (
        negative_returns.std()
        * np.sqrt(TRADING_DAYS)
    )

    # Sortino
    sortino = (
        returns.mean()
        * TRADING_DAYS
        / downside_deviation
        if downside_deviation != 0
        else np.nan
    )

    # Maximum Drawdown
    rolling_max = equity.cummax()

    drawdown = (
        equity / rolling_max
    ) - 1

    max_drawdown = drawdown.min()

    # Calmar
    calmar = (
        cagr / abs(max_drawdown)
        if max_drawdown != 0
        else np.nan
    )

    # Exposure
    average_exposure = (
        df["Position"].mean()
    )

    # Trades
    trades = (
        df["Trade"] > 0
    ).sum()

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualised Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Maximum Drawdown": max_drawdown,
        "Calmar Ratio": calmar,
        "Average Exposure": average_exposure,
        "Number of Trades": trades
    }


def calculate_buy_hold(df):

    equity = df[
        "Buy_Hold_Equity"
    ]

    total_return = (
        equity.iloc[-1] / equity.iloc[0]
    ) - 1

    years = len(df) / TRADING_DAYS

    cagr = (
        (equity.iloc[-1] / equity.iloc[0])
        ** (1 / years)
    ) - 1

    return {
        "Buy & Hold Total Return":
            total_return,
        "Buy & Hold CAGR":
            cagr
    }


def main():

    df = pd.read_csv(
        INPUT_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    metrics = calculate_metrics(df)

    benchmark = calculate_buy_hold(df)

    print("\n" + "=" * 60)
    print("FINAL PERFORMANCE REPORT")
    print("=" * 60)

    print("\nML STRATEGY")

    for name, value in metrics.items():

        if name == "Number of Trades":

            print(
                f"{name}: {value}"
            )

        else:

            print(
                f"{name}: {value:.4f}"
            )

    print("\nBUY & HOLD")

    for name, value in benchmark.items():

        print(
            f"{name}: {value:.4f}"
        )


if __name__ == "__main__":
    main()