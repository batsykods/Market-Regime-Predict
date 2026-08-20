import pandas as pd
import streamlit as st


PREDICTIONS_PATH = (
    "data/processed/walk_forward_results.csv"
)

BACKTEST_PATH = (
    "data/processed/risk_adjusted_backtest.csv"
)


st.set_page_config(
    page_title="Stock Regime & Signal Detection",
    layout="wide"
)


st.title(
    "Stock Market Regime & Signal Detection"
)


# Load data
predictions = pd.read_csv(
    PREDICTIONS_PATH
)

backtest = pd.read_csv(
    BACKTEST_PATH
)


predictions["Date"] = pd.to_datetime(
    predictions["Date"]
)

backtest["Date"] = pd.to_datetime(
    backtest["Date"]
)


# Latest prediction
latest = predictions.iloc[-1]


signal_map = {
    0: "HOLD",
    1: "BUY",
    2: "SELL"
}


signal = signal_map[
    int(latest["Prediction"])
]


# --------------------------------------------------
# Current Signal
# --------------------------------------------------

st.header("Current Market Signal")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Signal",
        signal
    )


with col2:

    st.metric(
        "Regime",
        int(latest["Regime"])
    )


with col3:

    st.metric(
        "Buy Probability",
        f"{latest['BUY_Probability']:.2%}"
    )


with col4:

    st.metric(
        "Sell Probability",
        f"{latest['SELL_Probability']:.2%}"
    )


# --------------------------------------------------
# Equity Curve
# --------------------------------------------------

st.header("Strategy Performance")


chart_data = backtest.set_index(
    "Date"
)[
    [
        "Strategy_Equity",
        "Buy_Hold_Equity"
    ]
]

st.line_chart(
    chart_data
)


# --------------------------------------------------
# Drawdown
# --------------------------------------------------

st.header("Strategy Drawdown")


rolling_max = (
    backtest["Strategy_Equity"]
    .cummax()
)

backtest["Drawdown"] = (
    backtest["Strategy_Equity"]
    / rolling_max
    - 1
)

drawdown_data = backtest.set_index(
    "Date"
)[
    ["Drawdown"]
]

st.line_chart(
    drawdown_data
)


# --------------------------------------------------
# Position Exposure
# --------------------------------------------------

st.header("Position Exposure")


exposure_data = backtest.set_index(
    "Date"
)[
    ["Position"]
]

st.line_chart(
    exposure_data
)


# --------------------------------------------------
# Recent Predictions
# --------------------------------------------------

st.header("Recent Predictions")


recent = predictions[
    [
        "Date",
        "Regime",
        "Prediction",
        "BUY_Probability",
        "HOLD_Probability",
        "SELL_Probability"
    ]
].tail(20).copy()


recent["Prediction"] = (
    recent["Prediction"]
    .map(signal_map)
)


st.dataframe(
    recent,
    use_container_width=True
)


st.caption(
    "Model outputs are historical research results "
    "and are not financial advice."
)