import pandas as pd
import matplotlib.pyplot as plt


PATH = "data/processed/nifty50_regimes.csv"


df = pd.read_csv(PATH)

df["Date"] = pd.to_datetime(df["Date"])


plt.figure(figsize=(14, 7))

for regime in sorted(df["Regime"].unique()):

    subset = df[df["Regime"] == regime]

    plt.scatter(
        subset["Date"],
        subset["Close"],
        s=8,
        label=f"Regime {regime}"
    )


plt.plot(
    df["Date"],
    df["Close"],
    alpha=0.4,
    linewidth=1
)

plt.title("NIFTY 50 Market Regimes")
plt.xlabel("Date")
plt.ylabel("NIFTY 50 Close")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()