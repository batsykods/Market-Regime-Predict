import pandas as pd
import matplotlib.pyplot as plt


PATH = "data/processed/backtest_results.csv"


df = pd.read_csv(PATH)

df["Date"] = pd.to_datetime(df["Date"])


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
    "ML Strategy vs NIFTY Buy & Hold"
)

plt.xlabel("Date")
plt.ylabel("Portfolio Value")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()