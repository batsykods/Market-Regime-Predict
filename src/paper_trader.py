import os
import pandas as pd
from src.live_signal import get_snapshot

LEDGER_PATH = "data/processed/paper_trades.csv"
CAPITAL = 100000.0


def main():
    snapshot = get_snapshot(CAPITAL)
    row = {
        "Timestamp": snapshot["timestamp"],
        "Price": snapshot["price"],
        "Signal": snapshot["signal"],
        "Regime": snapshot["regime"],
        "Predicted_Return": snapshot["predicted_return"],
        "Position_Size": snapshot["position_size"],
        "Target_Price": snapshot["target_price"],
        "Stop_Price": snapshot["stop_price"],
        "Expected_Profit": snapshot["expected_profit"],
        "Holding_Days": snapshot["holding_days"],
    }
    os.makedirs("data/processed", exist_ok=True)
    if os.path.exists(LEDGER_PATH):
        ledger = pd.read_csv(LEDGER_PATH)
        if not ledger.empty and str(ledger.iloc[-1]["Timestamp"]) == str(row["Timestamp"]):
            ledger.iloc[-1] = row
        else:
            ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
    else:
        ledger = pd.DataFrame([row])
    ledger.to_csv(LEDGER_PATH, index=False)
    print("PAPER TRADE SNAPSHOT SAVED")
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
