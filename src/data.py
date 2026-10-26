from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROVINCES = {
    "CN011":"北京","CN012":"天津","CN013":"河北","CN014":"山西","CN015":"内蒙古",
    "CN021":"辽宁","CN022":"吉林","CN023":"黑龙江","CN031":"上海","CN032":"江苏",
    "CN033":"浙江","CN034":"安徽","CN035":"福建","CN036":"江西","CN037":"山东",
    "CN041":"河南","CN042":"湖北","CN043":"湖南","CN044":"广东","CN045":"广西",
    "CN046":"海南","CN050":"重庆","CN051":"四川","CN052":"贵州","CN053":"云南",
    "CN054":"西藏","CN061":"陕西","CN062":"甘肃","CN063":"青海","CN064":"宁夏","CN065":"新疆"}

TIME_FILES = {"meteo":"meteo_maize_CN.csv", "ndvi":"ndvi_maize_CN.csv", "fpar":"fpar_maize_CN.csv", "moisture":"soil_moisture_maize_CN.csv"}
STATIC_FILES = {"soil":"soil_maize_CN.csv", "location":"location_maize_CN.csv", "calendar":"crop_calendar_maize_CN.csv", "mask":"crop_mask_maize_CN.csv"}

@dataclass
class DatasetBundle:
    dynamic: np.ndarray       # N,T,D weather + remote sensing + moisture
    soil: np.ndarray          # N,S
    coords: np.ndarray        # N,2
    y: np.ndarray             # N
    years: np.ndarray
    regions: np.ndarray
    region_index: np.ndarray  # maps examples to unique-region index
    region_coords: np.ndarray
    feature_names: List[str]
    soil_names: List[str]
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    scalers: Dict[str, StandardScaler]

def _read(root: Path, name: str) -> pd.DataFrame:
    p = root / name
    if not p.exists(): raise FileNotFoundError(f"Missing required CY-Bench file: {p}")
    return pd.read_csv(p)

def _date_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    return df.dropna(subset=["date"])

def _season_monthly(df: pd.DataFrame, calendar: pd.DataFrame, value_cols: List[str], steps: int) -> pd.DataFrame:
    """Aggregate each region-year only over its crop calendar, into equal-width phenology bins."""
    c = calendar.set_index("adm_id")[["sos", "eos"]].to_dict("index")
    rows = []
    for adm, g in df.groupby("adm_id"):
        if adm not in c: continue
        sos, eos = float(c[adm]["sos"]), float(c[adm]["eos"])
        for year, yg in g.groupby(g.date.dt.year):
            start = pd.Timestamp(year=int(year), month=1, day=1) + pd.Timedelta(days=sos-1)
            end_year = int(year) + int(eos < sos)
            end = pd.Timestamp(year=end_year, month=1, day=1) + pd.Timedelta(days=eos-1)
            s = yg[(yg.date >= start) & (yg.date <= end)].copy()
            if s.empty: continue
            width = max((end-start).days + 1, 1)
            s["bin"] = np.minimum(((s.date-start).dt.days * steps / width).astype(int), steps-1)
            a = s.groupby("bin")[value_cols].mean().reindex(range(steps)).interpolate(limit_direction="both")
            if a.isna().any().any(): continue
            a["adm_id"], a["year"] = adm, year
            rows.append(a.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def load_bundle(root: str | Path, steps: int, train_end: int, val_end: int) -> DatasetBundle:
    root = Path(root)
    cal = _read(root, STATIC_FILES["calendar"])
    y = _read(root, "yield_maize_CN.csv")[["adm_id","harvest_year","yield"]].dropna().rename(columns={"harvest_year":"year"})
    y["year"] = y.year.astype(int)
    pieces, names = [], []
    for key, fn in TIME_FILES.items():
        d = _date_col(_read(root, fn)); vals = [x for x in d.columns if x not in {"crop_name","adm_id","date"}]
        a = _season_monthly(d, cal, vals, steps)
        if a.empty: raise ValueError(f"No valid growing-season observations in {fn}")
        a = a.rename(columns={v:f"{key}_{v}" for v in vals}); pieces.append(a); names += [f"{key}_{v}" for v in vals]
    seq = pieces[0]
    for p in pieces[1:]: seq = seq.merge(p, on=["adm_id","year","bin"], how="inner")
    # Match label to the year in which the growing season ends; this is exact for same-year seasons.
    records = y.merge(seq, on=["adm_id","year"], how="inner")
    stat = _read(root, STATIC_FILES["soil"]).merge(_read(root, STATIC_FILES["location"]), on=["crop_name","adm_id"])
    stat = stat.merge(_read(root, STATIC_FILES["mask"])[["adm_id","crop_area","crop_area_percentage"]], on="adm_id", how="left")
    soil_cols = [c for c in stat.columns if c not in {"crop_name","adm_id","latitude","longitude"}]
    records = records.merge(stat[["adm_id","latitude","longitude"]+soil_cols], on="adm_id", how="inner")
    grouped = []
    for (adm, year), g in records.groupby(["adm_id","year"], sort=True):
        g = g.sort_values("bin")
        if len(g) == steps: grouped.append((adm,year,g))
    if not grouped: raise ValueError("No complete region-year sequences after joining labels and predictors.")
    regions = np.array([a for a,_,_ in grouped]); years = np.array([b for _,b,_ in grouped]);
    dyn = np.stack([g[names].to_numpy(float) for _,_,g in grouped]);
    soil = np.stack([g.iloc[0][soil_cols].to_numpy(float) for _,_,g in grouped]);
    coords = np.stack([g.iloc[0][["latitude","longitude"]].to_numpy(float) for _,_,g in grouped]);
    target = np.array([g.iloc[0]["yield"] for _,_,g in grouped], dtype=float)
    train, val, test = years <= train_end, (years > train_end) & (years <= val_end), years > val_end
    if not train.any() or not val.any() or not test.any(): raise ValueError("Time split produced an empty partition; adjust train_end_year/val_end_year.")
    ds, ss, ys = StandardScaler(), StandardScaler(), StandardScaler()
    ds.fit(dyn[train].reshape(-1,dyn.shape[-1])); dyn = ds.transform(dyn.reshape(-1,dyn.shape[-1])).reshape(dyn.shape)
    ss.fit(soil[train]); soil=ss.transform(soil); ys.fit(target[train,None]); target=ys.transform(target[:,None]).ravel()
    unique = np.array(sorted(set(regions))); lookup={x:i for i,x in enumerate(unique)}; ri=np.array([lookup[x] for x in regions])
    ucoords=np.stack([coords[np.where(regions==r)[0][0]] for r in unique])
    return DatasetBundle(dyn,soil,coords,target,years,regions,ri,ucoords,names,soil_cols,train,val,test,{"dynamic":ds,"soil":ss,"target":ys})
