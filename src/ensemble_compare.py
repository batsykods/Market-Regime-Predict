import pandas as pd
import numpy as np

ENSEMBLE="data/processed/walk_forward_regime_ensemble.csv"
BASE="data/processed/walk_forward_results.csv"


def report(path,name):
    d=pd.read_csv(path).replace([np.inf,-np.inf],np.nan).dropna(subset=["Predicted_Return","Future_Return"])
    y=d.Future_Return; p=d.Predicted_Return
    q=pd.qcut(p,10,labels=False,duplicates="drop")
    dec=pd.DataFrame({"p":p,"y":y,"q":q}).groupby("q").y.mean()
    return {"Model":name,"Samples":len(d),"MAE":(y-p).abs().mean(),"RMSE":np.sqrt(((y-p)**2).mean()),"Direction":(np.sign(y)==np.sign(p)).mean(),"Pearson":y.corr(p),"Spearman":y.corr(p,method="spearman"),"TopBottom":dec.iloc[-1]-dec.iloc[0]}


def main():
    out=pd.DataFrame([report(BASE,"Baseline"),report(ENSEMBLE,"Regime Ensemble")])
    print("\n"+"="*72); print("BASELINE VS REGIME-ENSEMBLE WALK-FORWARD COMPARISON"); print("="*72)
    print(out.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    print("\nPromotion rule: ensemble must improve ranking metrics without materially degrading error metrics.")

if __name__=="__main__": main()
