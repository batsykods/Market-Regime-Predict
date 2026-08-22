import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT_PATH = "data/processed/walk_forward_regimes.csv"
OUTPUT_PATH = "data/processed/walk_forward_regime_ensemble.csv"
FEATURES = ["Log_Return","Volatility_20","Volatility_5","Volatility_Ratio","SMA_20","SMA_50","Price_SMA20_Ratio","Price_SMA50_Ratio","Momentum_5","Momentum_20","Momentum_60","Volume_Change","Drawdown","ATR_14","Regime"]
HORIZON = 5
INITIAL_FRACTION = 0.70
MIN_REGIME_SAMPLES = 80


def prepare(df):
    df = df.copy()
    close = df["Close"].astype(float)
    lr = df["Log_Return"].astype(float)
    df["Volatility_5"] = lr.rolling(5).std() * np.sqrt(252)
    df["Volatility_Ratio"] = df["Volatility_5"] / df["Volatility_20"].replace(0, np.nan)
    df["Price_SMA20_Ratio"] = close / df["SMA_20"] - 1
    df["Price_SMA50_Ratio"] = close / df["SMA_50"] - 1
    df["Momentum_5"] = close.pct_change(5)
    df["Momentum_60"] = close.pct_change(60)
    df["Future_Return"] = close.shift(-HORIZON) / close - 1
    return df.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+["Future_Return"]).reset_index(drop=True)


def model():
    return XGBRegressor(n_estimators=180,max_depth=2,learning_rate=0.025,min_child_weight=10,subsample=.75,colsample_bytree=.75,reg_alpha=.25,reg_lambda=3,objective="reg:squarederror",eval_metric="rmse",random_state=42)


def fit_predict(train, row):
    global_model = model().fit(train[FEATURES].astype(float), train["Future_Return"].astype(float))
    gp = float(global_model.predict(row[FEATURES].astype(float))[0])
    regime = int(row["Regime"].iloc[0])
    rt = train[train["Regime"] == regime]
    if len(rt) >= MIN_REGIME_SAMPLES:
        rm = model().fit(rt[FEATURES].astype(float), rt["Future_Return"].astype(float))
        rp = float(rm.predict(row[FEATURES].astype(float))[0])
        # Blend conservatively; regime model cannot completely override the global model.
        return 0.70 * gp + 0.30 * rp, gp, rp, len(rt)
    return gp, gp, np.nan, len(rt)


def main():
    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = prepare(df.sort_values("Date").reset_index(drop=True))
    initial = int(len(df) * INITIAL_FRACTION)
    out=[]
    for i in range(initial,len(df)):
        train=df.iloc[:i]; row=df.iloc[[i]]
        pred,gp,rp,ns=fit_predict(train,row)
        out.append({"Date":row["Date"].iloc[0],"Close":row["Close"].iloc[0],"Future_Return":row["Future_Return"].iloc[0],"Predicted_Return":pred,"Global_Prediction":gp,"Regime_Prediction":rp,"Regime_Samples":ns,"Regime":int(row["Regime"].iloc[0]),"Volatility_20":float(row["Volatility_20"].iloc[0])})
        if (i-initial+1)%25==0: print(f"Processed {i-initial+1}/{len(df)-initial}")
    r=pd.DataFrame(out); os.makedirs("data/processed",exist_ok=True); r.to_csv(OUTPUT_PATH,index=False)
    y,p=r.Future_Return,r.Predicted_Return
    print("\n"+"="*64); print("WALK-FORWARD REGIME-ADAPTIVE ENSEMBLE COMPLETE"); print("="*64)
    print(f"Saved to: {OUTPUT_PATH}"); print(f"Predictions generated: {len(r)}")
    print(f"MAE: {mean_absolute_error(y,p):.6f}"); print(f"RMSE: {np.sqrt(mean_squared_error(y,p)):.6f}"); print(f"Direction Accuracy: {np.mean(np.sign(y)==np.sign(p)):.4f}"); print(f"Pearson Correlation: {y.corr(p):.4f}"); print(f"Spearman Correlation: {y.corr(p,method='spearman'):.4f}")

if __name__ == "__main__": main()
