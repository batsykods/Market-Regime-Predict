import numpy as np
import pandas as pd


TRADING_DAYS = 252


def calculate_metrics(df):

    returns = (
        df["Strategy_Return_After_Cost"]
        .dropna()
    )

    equity = df["Strategy_Equity"]

    total_return = (
        equity.iloc[-1]
        / equity.iloc[0]
        - 1
    )

    years = (
        len(returns)
        / TRADING_DAYS
    )

    cagr = (
        (equity.iloc[-1] / equity.iloc[0])
        ** (1 / years)
        - 1
    )

    annualised_volatility = (
        returns.std()
        * np.sqrt(TRADING_DAYS)
    )

    sharpe = (
        returns.mean()
        / returns.std()
        * np.sqrt(TRADING_DAYS)
    )

    rolling_max = equity.cummax()

    drawdown = (
        equity / rolling_max
        - 1
    )

    max_drawdown = drawdown.min()

    win_rate = (
        (returns > 0).sum()
        / len(returns)
    )

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualised Volatility":
            annualised_volatility,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown":
            max_drawdown,
        "Win Rate": win_rate
    }


if __name__ == "__main__":

    df = pd.read_csv(
        "data/processed/backtest_results.csv"
    )

    metrics = calculate_metrics(df)

    print("\nBACKTEST PERFORMANCE")
    print("=" * 50)

    for name, value in metrics.items():

        print(
            f"{name}: {value:.4f}"
        )