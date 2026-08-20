import pandas as pd
import matplotlib.pyplot as plt


INPUT_PATH = "data/processed/risk_adjusted_backtest.csv"


def main():

    df = pd.read_csv(INPUT_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    # Equity curve
    plt.figure(figsize=(14, 7))

    plt.plot(
        df["Date"],
        df["Strategy_Equity"],
        label="ML Strategy"
    )

    plt.plot(
        df["Date"],
        df["Buy_Hold_Equity"],
        label="Buy & Hold"
    )

    plt.title(
        "ML Strategy vs Buy & Hold"
    )

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.show()


    # Drawdown
    rolling_max = (
        df["Strategy_Equity"]
        .cummax()
    )

    drawdown = (
        df["Strategy_Equity"]
        / rolling_max
        - 1
    )

    plt.figure(figsize=(14, 5))

    plt.plot(
        df["Date"],
        drawdown
    )

    plt.title(
        "Strategy Drawdown"
    )

    plt.xlabel("Date")
    plt.ylabel("Drawdown")

    plt.grid(True)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()