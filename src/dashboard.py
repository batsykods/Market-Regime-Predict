import os
import sys
import pandas as pd
import streamlit as st

# Make repository root importable when Streamlit executes src/dashboard.py directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.live_signal import get_snapshot

st.set_page_config(page_title="NIFTY ML Signal Dashboard", layout="wide")
st.title("NIFTY 50 ML Signal Dashboard")
st.caption("Daily-bar live inference. Paper trading only; no broker orders are placed.")

@st.cache_data(ttl=300)
def load_snapshot():
    return get_snapshot()

snapshot = load_snapshot()

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Signal", snapshot["signal"])
with c2: st.metric("NIFTY", f"₹{snapshot['price']:,.2f}")
with c3: st.metric("Regime", str(snapshot["regime"]))
with c4: st.metric("5D Expected Return", f"{snapshot['predicted_return']:.2%}")

a, b, c, d = st.columns(4)
with a: st.metric("Position", f"{snapshot['position_size']:.2%}")
with b: st.metric("Target", f"₹{snapshot['target_price']:,.2f}")
with c: st.metric("Stop", f"₹{snapshot['stop_price']:,.2f}")
with d: st.metric("Expected Profit", f"₹{snapshot['expected_profit']:,.2f}")

st.write(f"Prediction percentile: **{snapshot['prediction_percentile']:.1%}**")
st.write(f"Risk ratio: **{snapshot['risk_ratio']:.3f}** | Risk cutoff: **{snapshot['risk_cutoff']:.3f}**")
st.write(f"Signal cutoff: **{snapshot['prediction_cutoff']:.3%}** | Holding window: **{snapshot['holding_days']} trading days**")
st.write(f"Volatility (20D): **{snapshot['volatility_20']:.2%}** | ATR(14): **₹{snapshot['atr_14']:,.2f}**")
st.write(f"Latest market bar: **{snapshot['timestamp']}**")

st.divider()
st.subheader("Backtest Performance")
try:
    backtest = pd.read_csv("data/processed/risk_adjusted_backtest.csv")
    backtest["Date"] = pd.to_datetime(backtest["Date"])
    cols = [c for c in ["Strategy_Equity", "Buy_Hold_Equity"] if c in backtest.columns]
    if cols:
        st.line_chart(backtest.set_index("Date")[cols])
except FileNotFoundError:
    st.info("Run the backtest pipeline first.")

st.subheader("Paper Trade Ledger")
try:
    ledger = pd.read_csv("data/processed/paper_trades.csv")
    st.dataframe(ledger.tail(25), use_container_width=True)
except FileNotFoundError:
    st.info("No paper-trade snapshots yet. Run: python -m src.paper_trader")

st.caption("Research/paper-trading system. Predictions are estimates, not guaranteed returns or financial advice.")
