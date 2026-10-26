from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd, seaborn as sns

def main():
    p=argparse.ArgumentParser(); p.add_argument("--artifacts",default="artifacts"); a=p.parse_args(); root=Path(a.artifacts); root.mkdir(exist_ok=True)
    metrics=pd.concat([pd.read_csv(x) for x in root.glob("metrics_*.csv")],ignore_index=True)
    metrics.to_csv(root/"table_model_comparison.csv",index=False)
    sns.set_theme(style="whitegrid",font_scale=1.05)
    fig,ax=plt.subplots(figsize=(8,4)); sns.barplot(metrics,x="experiment",y="rmse",ax=ax,color="#4776b4"); ax.tick_params(axis="x",rotation=35); ax.set(title="Test RMSE comparison",ylabel="RMSE (t ha$^{-1}$)",xlabel=""); fig.tight_layout(); fig.savefig(root/"fig_model_comparison_rmse.png",dpi=300); plt.close(fig)
    for f in root.glob("predictions_*.csv"):
        d=pd.read_csv(f); d=d[d.split=="test"]; stem=f.stem.replace("predictions_","")
        fig,ax=plt.subplots(figsize=(5,5)); sns.scatterplot(d,x="actual_yield",y="predicted_yield",hue="predicted_risk",ax=ax,palette={"high":"#c73e1d","medium":"#e3a008","low":"#238636"}); lo=min(d.actual_yield.min(),d.predicted_yield.min()); hi=max(d.actual_yield.max(),d.predicted_yield.max()); ax.plot([lo,hi],[lo,hi],"k--",lw=1); ax.set(title=f"Test prediction: {stem}",xlabel="Observed yield (t ha$^{-1}$)",ylabel="Predicted yield (t ha$^{-1}$)"); fig.tight_layout(); fig.savefig(root/f"fig_scatter_{stem}.png",dpi=300); plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4)); d["error"]=d.predicted_yield-d.actual_yield; sns.histplot(d,x="error",hue="predicted_risk",multiple="stack",ax=ax,palette={"high":"#c73e1d","medium":"#e3a008","low":"#238636"}); ax.set(title=f"Test error distribution: {stem}",xlabel="Prediction error (t ha$^{-1}$)"); fig.tight_layout(); fig.savefig(root/f"fig_error_{stem}.png",dpi=300); plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,5)); sns.scatterplot(d,x="longitude",y="latitude",size="actual_yield",hue="predicted_risk",sizes=(40,250),palette={"high":"#c73e1d","medium":"#e3a008","low":"#238636"},ax=ax); ax.set(title=f"Spatial risk map: {stem}",xlabel="Longitude",ylabel="Latitude"); fig.tight_layout(); fig.savefig(root/f"fig_risk_map_{stem}.png",dpi=300); plt.close(fig)
    # Lag parameters are directly thesis-ready evidence for the proposed model.
    prop=metrics[metrics.experiment=="proposed"]
    if not prop.empty and prop.lag_months.notna().all():
        vals=json.loads(prop.iloc[-1].lag_months)
        if vals:
            fig,ax=plt.subplots(figsize=(6,4)); ax.bar(range(1,len(vals)+1),vals,color="#7f3c8d"); ax.set(title="Learned physiological lag by attention head",xlabel="Attention head",ylabel="Lag (months)"); fig.tight_layout(); fig.savefig(root/"fig_learned_lags.png",dpi=300); plt.close(fig)
if __name__=="__main__": main()
