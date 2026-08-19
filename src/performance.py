import numpy as np
import pandas as pd


def calculate_metrics(df):

    returns = df["Strategy_Return"].dropna()

    equity = (1 + returns).cumprod()

    total_return = equity.iloc[-1] - 1

    years = len(returns) / 252

    cagr = (
        equity.iloc[-1] ** (1 / years)
        - 1
    )

    volatility = (
        returns.std() * np.sqrt(252)
    )

    sharpe = (
        returns.mean()
        / returns.std()
        * np.sqrt(252)
    )

    downside = returns[returns < 0]

    sortino = (
        returns.mean()
        / downside.std()
        * np.sqrt(252)
    )

    drawdown = (
        equity /
        equity.cummax()
        - 1
    )

    max_drawdown = drawdown.min()

    winning = returns[returns > 0]
    losing = returns[returns < 0]

    win_rate = (
        len(winning) / len(returns)
    )

    profit_factor = (
        winning.sum()
        / abs(losing.sum())
        if len(losing) > 0
        else np.inf
    )

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor
    }