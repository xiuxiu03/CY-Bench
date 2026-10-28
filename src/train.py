from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from .data import load_bundle
from .llm import cached_knowledge_embeddings
from .models import LSTMRegressor, TransformerRegressor, ProposedModel

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def metrics(y,p): return {"rmse":float(mean_squared_error(y,p)**.5),"r2":float(r2_score(y,p))}
def year_batches(bundle,mask):
    for y in sorted(set(bundle.years[mask])): yield np.where(mask & (bundle.years==y))[0]
def seasonal_features(bundle):
    return np.concatenate([bundle.dynamic.mean(1),bundle.dynamic.std(1),bundle.dynamic.max(1),bundle.soil],axis=1)
def inverse(bundle,x): return bundle.scalers["target"].inverse_transform(np.asarray(x).reshape(-1,1)).ravel()

def main():
    pa=argparse.ArgumentParser(); pa.add_argument("--config",default="configs/default.yaml"); pa.add_argument("--data-dir"); pa.add_argument("--experiment",required=True,choices=["xgboost","lstm","transformer","proposed","ablate_no_lag","ablate_no_llm"]); pa.add_argument("--use-llm",action="store_true")
    a=pa.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text());
    if a.data_dir: cfg["data_dir"]=a.data_dir
    seed_all(cfg["seed"]); out=Path(cfg["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    b=load_bundle(cfg["data_dir"],cfg["sequence_steps"],cfg["train_end_year"],cfg["val_end_year"])
    use_llm=bool(a.use_llm or cfg.get("use_llm",False)) and a.experiment not in {"ablate_no_llm"}
    if a.experiment=="xgboost":
        x=seasonal_features(b); model=XGBRegressor(n_estimators=500,max_depth=3,learning_rate=.03,subsample=.8,colsample_bytree=.8,random_state=cfg["seed"],n_jobs=-1,objective="reg:squarederror")
        model.fit(x[b.train_mask],b.y[b.train_mask],eval_set=[(x[b.val_mask],b.y[b.val_mask])],verbose=False); pred=model.predict(x); tau=[]
    else:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        knowledge=None
        if use_llm:
            k_matrix = cached_knowledge_embeddings(b, cfg["llm_cache"])
            # 确保读进来的是 numpy 数组
            if isinstance(k_matrix, list):
                k_matrix = np.array(k_matrix)
            # 安全提取对应的知识向量
            knowledge = k_matrix[b.region_index]
            
            # Scale embeddings on train only, avoiding test-region/year influence.
            mu, sd = knowledge[b.train_mask].mean(0), knowledge[b.train_mask].std(0) + 1e-6
            knowledge = (knowledge - mu) / sd
        weather=[i for i,n in enumerate(b.feature_names) if n.startswith(("meteo_","moisture_"))]; remote=[i for i,n in enumerate(b.feature_names) if n.startswith(("ndvi_","fpar_"))]
        if a.experiment=="lstm": model=LSTMRegressor(b.dynamic.shape[-1],b.soil.shape[-1],cfg["hidden_dim"],cfg["dropout"])
        elif a.experiment=="transformer": model=TransformerRegressor(b.dynamic.shape[-1],b.soil.shape[-1],cfg["hidden_dim"],cfg["num_heads"],cfg["dropout"])
        else: model=ProposedModel(b.dynamic.shape[-1],b.soil.shape[-1],weather,remote,0 if knowledge is None else knowledge.shape[1],cfg["llm_dim"],cfg["num_heads"],cfg["dropout"],cfg["lag_min"],cfg["lag_max"])
        model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"]); loss_fn=torch.nn.MSELoss(); best=(float("inf"),None); wait=0
        def predict(mask,train=False):
            result=[]; model.train(train)
            for ix in year_batches(b,mask):
                xx=torch.tensor(b.dynamic[ix],dtype=torch.float32,device=device); ss=torch.tensor(b.soil[ix],dtype=torch.float32,device=device)
                kk=None if knowledge is None else torch.tensor(knowledge[ix],dtype=torch.float32,device=device)
                enable_lag=a.experiment!="ablate_no_lag"
                with torch.set_grad_enabled(train):
                    z=model(xx,ss,kk,enable_lag) if isinstance(model,ProposedModel) else model(xx,ss,kk)
                    if train:
                        opt.zero_grad(); loss=loss_fn(z,torch.tensor(b.y[ix],dtype=torch.float32,device=device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                result.extend(zip(ix,z.detach().cpu().numpy()))
            ans=np.zeros(len(b.y)); [ans.__setitem__(i,v) for i,v in result]; return ans
        for ep in range(cfg["epochs"]):
            _=predict(b.train_mask,True); val=predict(b.val_mask); score=mean_squared_error(b.y[b.val_mask],val[b.val_mask])
            if score<best[0]: best=(score,{k:v.detach().cpu() for k,v in model.state_dict().items()}); wait=0
            else: wait+=1
            if wait>=cfg["patience"]: break
        model.load_state_dict(best[1]); pred=predict(np.ones(len(b.y),dtype=bool)); torch.save(model.state_dict(),out/f"{a.experiment}.pt")
        tau=[] if not isinstance(model,ProposedModel) else model.last_tau.cpu().tolist()
    actual=inverse(b,b.y); predicted=inverse(b,pred); q5,q10=np.quantile(actual[b.train_mask],[.05,.10])
    risk=lambda z: np.where(z<q5,"high",np.where(z<q10,"medium","low"))
    frame=pd.DataFrame({"adm_id":b.regions,"year":b.years,"split":np.where(b.train_mask,"train",np.where(b.val_mask,"val","test")),"actual_yield":actual,"predicted_yield":predicted,"actual_risk":risk(actual),"predicted_risk":risk(predicted),"latitude":b.coords[:,0],"longitude":b.coords[:,1]})
    frame.to_csv(out/f"predictions_{a.experiment}.csv",index=False)
    row={"experiment":a.experiment,"use_llm":use_llm,"n":int(b.test_mask.sum()),**metrics(actual[b.test_mask],predicted[b.test_mask]),"risk_accuracy":float((frame.loc[b.test_mask,"actual_risk"]==frame.loc[b.test_mask,"predicted_risk"]).mean()),"risk_q5":float(q5),"risk_q10":float(q10),"lag_months":json.dumps(tau)}
    pd.DataFrame([row]).to_csv(out/f"metrics_{a.experiment}.csv",index=False); print(json.dumps(row,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
