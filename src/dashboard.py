import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


DATA_PATH = "data/processed/nifty50_signal_data.csv"


st.set_page_config(
    page_title="Stock Regime & Signal Detection",
    layout="wide"
)


# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna().reset_index(drop=True)


# -----------------------------
# Basic calculations
# -----------------------------

# Prototype signal
df["Signal"] = df["Target"].map({
    0: "HOLD",
    1: "BUY",
    2: "SELL"
})


# Simple strategy return
df["Position"] = df["Target"].map({
    0: 0,
    1: 1,
    2: -1
})

df["Future_Return"] = (
    df["Close"].shift(-1) /
    df["Close"]
) - 1

df["Strategy_Return"] = (
    df["Position"] *
    df["Future_Return"]
)

df = df.dropna(
    subset=["Strategy_Return"]
).reset_index(drop=True)


# Equity curve
df["Strategy_Equity"] = (
    1 + df["Strategy_Return"]
).cumprod()

df["BuyHold_Equity"] = (
    1 + df["Future_Return"]
).cumprod()


# Drawdown
df["Drawdown"] = (
    df["Strategy_Equity"] /
    df["Strategy_Equity"].cummax()
) - 1


# Metrics
total_return = (
    df["Strategy_Equity"].iloc[-1] - 1
)

years = len(df) / 252

cagr = (
    df["Strategy_Equity"].iloc[-1]
    ** (1 / years)
) - 1

volatility = (
    df["Strategy_Return"].std()
    * np.sqrt(252)
)

sharpe = (
    df["Strategy_Return"].mean()
    / df["Strategy_Return"].std()
    * np.sqrt(252)
)

max_drawdown = df["Drawdown"].min()

winning = df[
    df["Strategy_Return"] > 0
]

win_rate = (
    len(winning) /
    len(df)
)

buyhold_return = (
    df["BuyHold_Equity"].iloc[-1] - 1
)


# -----------------------------
# Regime labels
# -----------------------------

REGIME_LABELS = {
    0: "SIDEWAYS",
    1: "BULL",
    2: "BEAR"
}


# -----------------------------
# Dashboard
# -----------------------------

st.title(
    "Stock Market Regime & Signal Detection"
)

st.caption(
    "NIFTY 50 Machine Learning Trading System"
)


# -----------------------------
# Latest state
# -----------------------------

latest = df.iloc[-1]

regime = REGIME_LABELS.get(
    int(latest["Regime"]),
    f"REGIME {int(latest['Regime'])}"
)

signal = latest["Signal"]


st.subheader("Current Market State")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "NIFTY 50",
        f"{latest['Close']:,.2f}"
    )

with col2:
    st.metric(
        "Market Regime",
        regime
    )

with col3:
    st.metric(
        "Signal",
        signal
    )

with col4:
    st.metric(
        "Momentum",
        f"{latest['Momentum_20']:.2%}"
    )


# -----------------------------
# Performance
# -----------------------------

st.subheader("Strategy Performance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "CAGR",
        f"{cagr:.2%}"
    )

with col2:
    st.metric(
        "Total Return",
        f"{total_return:.2%}"
    )

with col3:
    st.metric(
        "Sharpe",
        f"{sharpe:.2f}"
    )

with col4:
    st.metric(
        "Max Drawdown",
        f"{max_drawdown:.2%}"
    )

with col5:
    st.metric(
        "Win Rate",
        f"{win_rate:.2%}"
    )


# -----------------------------
# Price
# -----------------------------

st.subheader("NIFTY 50 Price")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Close"],
        name="NIFTY 50"
    )
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# -----------------------------
# Equity Curve
# -----------------------------

st.subheader("Strategy vs Buy & Hold")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Strategy_Equity"],
        name="ML Strategy"
    )
)

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["BuyHold_Equity"],
        name="Buy & Hold"
    )
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Growth of ₹1",
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# -----------------------------
# Drawdown
# -----------------------------

st.subheader("Strategy Drawdown")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Drawdown"],
        name="Drawdown"
    )
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Drawdown",
    height=400
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# -----------------------------
# Regime Distribution
# -----------------------------

st.subheader("Market Regime Distribution")

regime_counts = (
    df["Regime"]
    .map(REGIME_LABELS)
    .value_counts()
)

st.bar_chart(regime_counts)


# -----------------------------
# Recent signals
# -----------------------------

st.subheader("Recent Signals")

recent = df[
    [
        "Date",
        "Close",
        "Regime",
        "Signal",
        "Momentum_20",
        "Volatility_20"
    ]
].tail(20).copy()

recent["Regime"] = recent[
    "Regime"
].map(REGIME_LABELS)

st.dataframe(
    recent,
    use_container_width=True
)