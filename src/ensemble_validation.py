import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

INPUT = "data/processed/walk_forward_regimes.csv"
OUT = "data/processed/ensemble_validation.csv"
FEATURES = ["Log_Return","Volatility_20","Volatility_5","Volatility_Ratio","SMA_20","SMA_50","Price_SMA20_Ratio","Price_SMA50_Ratio","Momentum_5","Momentum_20","Momentum_60","Volume_Change","Drawdown","ATR_14","Regime"]
H = 5
MIN_REGIME = 80
BLEND = .30


def prep(df):
    df=df.copy().sort_values("Date").reset_index(drop=True)
    c=df["Close"].astype(float); lr=df["Log_Return"].astype(float)
    df["Volatility_5"]=lr.rolling(5).std()*np.sqrt(252)
    df["Volatility_Ratio"]=df["Volatility_5"]/df["Volatility_20"].replace(0,np.nan)
    df["Price_SMA20_Ratio"]=c/df["SMA_20"]-1; df["Price_SMA50_Ratio"]=c/df["SMA_50"]-1
    df["Momentum_5"]=c.pct_change(5); df["Momentum_60"]=c.pct_change(60)
    df["Future_Return"]=c.shift(-H)/c-1
    return df.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+["Future_Return"]).reset_index(drop=True)


def make_model():
    return XGBRegressor(n_estimators=180,max_depth=2,learning_rate=.025,min_child_weight=10,subsample=.75,colsample_bytree=.75,reg_alpha=.25,reg_lambda=3,objective="reg:squarederror",eval_metric="rmse",random_state=42)


def predict(train,row):
    gm=make_model().fit(train[FEATURES],train["Future_Return"])
    gp=float(gm.predict(row[FEATURES])[0]); reg=int(row["Regime"].iloc[0]); rt=train[train["Regime"]==reg]
    if len(rt)>=MIN_REGIME:
        rm=make_model().fit(rt[FEATURES],rt["Future_Return"]); rp=float(rm.predict(row[FEATURES])[0])
        return (1-BLEND)*gp+BLEND*rp
    return gp


def metrics(x):
    y=x.Future_Return; p=x.Predicted_Return
    return {"samples":len(x),"return":(x.Realised_Strategy_Return+1).prod()-1,"buy_hold":(1+y).prod()-1,"outperformance":(x.Realised_Strategy_Return+1).prod()/(1+y).prod()-1,"mae":(y-p).abs().mean(),"rmse":np.sqrt(((y-p)**2).mean()),"direction":(np.sign(y)==np.sign(p)).mean(),"pearson":y.corr(p),"spearman":y.corr(p,method="spearman")}


def main():
    df=pd.read_csv(INPUT); df["Date"]=pd.to_datetime(df["Date"]); df=prep(df)
    dev_end=pd.Timestamp("2025-12-23"); test=df[df.Date>dev_end].copy().reset_index(drop=True)
    dev=df[df.Date<=dev_end].copy(); rows=[]
    print("\n"+"="*72); print("FROZEN OUT-OF-SAMPLE REGIME-ENSEMBLE VALIDATION"); print("="*72)
    for i in range(len(test)):
        train=dev if i==0 else pd.concat([dev,test.iloc[:i]],ignore_index=True)
        row=test.iloc[[i]].copy(); pred=predict(train,row)
        rows.append({"Date":row.Date.iloc[0],"Close":row.Close.iloc[0],"Regime":int(row.Regime.iloc[0]),"Predicted_Return":pred,"Future_Return":row.Future_Return.iloc[0]})
    r=pd.DataFrame(rows); r["Realised_Strategy_Return"]=np.where(r.Predicted_Return>r.Predicted_Return.quantile(.80),r.Future_Return,0.0)
    os.makedirs("data/processed",exist_ok=True); r.to_csv(OUT,index=False)
    m=metrics(r)
    for k,v in m.items(): print(f"{k}: {v:.6f}" if isinstance(v,float) else f"{k}: {v}")
    print("Saved:",OUT)
    print("\nDECISION: compare these frozen metrics against the original model before promotion.")

if __name__=="__main__": main()
