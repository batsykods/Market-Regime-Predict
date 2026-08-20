from fastapi import FastAPI
import pandas as pd


app = FastAPI(
    title="Stock Regime & Signal API",
    version="1.0.0"
)


PREDICTIONS_PATH = (
    "data/processed/walk_forward_results.csv"
)

BACKTEST_PATH = (
    "data/processed/risk_adjusted_backtest.csv"
)


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.get("/latest-signal")
def latest_signal():

    df = pd.read_csv(
        PREDICTIONS_PATH
    )

    latest = df.iloc[-1]

    prediction_map = {
        0: "HOLD",
        1: "BUY",
        2: "SELL"
    }

    prediction = int(
        latest["Prediction"]
    )

    return {
        "date": str(
            latest["Date"]
        ),
        "signal": prediction_map[
            prediction
        ],
        "regime": int(
            latest["Regime"]
        ),
        "buy_probability": float(
            latest["BUY_Probability"]
        ),
        "hold_probability": float(
            latest["HOLD_Probability"]
        ),
        "sell_probability": float(
            latest["SELL_Probability"]
        )
    }


@app.get("/performance")
def performance():

    df = pd.read_csv(
        BACKTEST_PATH
    )

    initial_value = (
        df["Strategy_Equity"].iloc[0]
    )

    final_value = (
        df["Strategy_Equity"].iloc[-1]
    )

    total_return = (
        final_value / initial_value
    ) - 1

    return {
        "initial_capital": initial_value,
        "final_value": final_value,
        "total_return": total_return,
        "average_exposure": float(
            df["Position"].mean()
        )
    }