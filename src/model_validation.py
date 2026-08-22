import numpy as np
import pandas as pd

INPUT = "data/processed/walk_forward_regime_ensemble.csv"

def main():
    df=pd.read_csv(INPUT).replace([np.inf,-np.inf],np.nan).dropna(subset=["Predicted_Return","Future_Return","Regime"])
    err=df.Future_Return-df.Predicted_Return
    correct=np.sign(df.Future_Return)==np.sign(df.Predicted_Return)
    print("\n"+"="*72); print("REGIME-ENSEMBLE WALK-FORWARD VALIDATION"); print("="*72)
    print(f"Samples: {len(df)}"); print(f"MAE: {err.abs().mean():.6f}"); print(f"RMSE: {np.sqrt((err**2).mean()):.6f}"); print(f"Direction accuracy: {correct.mean():.4%}"); print(f"Pearson correlation: {df.Predicted_Return.corr(df.Future_Return):.4f}"); print(f"Spearman correlation: {df.Predicted_Return.corr(df.Future_Return,method='spearman'):.4f}")
    out=df.assign(Correct=correct,Error=err).groupby("Regime").agg(Samples=("Future_Return","size"),MAE=("Error",lambda x:x.abs().mean()),RMSE=("Error",lambda x:np.sqrt((x**2).mean())),Direction_Accuracy=("Correct","mean"),Prediction_Correlation=("Predicted_Return",lambda x:x.corr(df.loc[x.index,"Future_Return"])),Mean_Actual_Return=("Future_Return","mean"),Mean_Predicted_Return=("Predicted_Return","mean"))
    print("\nREGIME VALIDATION"); print(out.to_string(float_format=lambda x:f"{x:.6f}"))
    corr=df.Predicted_Return.corr(df.Future_Return,method="spearman")
    print("\nMODEL DECISION")
    print("KEEP: meaningful ranking signal; proceed to independent validation." if corr>=.20 else "WEAK: limited ranking signal; do not promote yet.")
if __name__=="__main__": main()
