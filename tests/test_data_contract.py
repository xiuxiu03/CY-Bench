"""Server-side smoke tests: run after pip install -r requirements.txt."""
from pathlib import Path
from src.data import load_bundle

def test_cn_data_contract():
    root=Path(__file__).parents[2] / "data" / "CN"
    if not root.exists(): return  # CI can mount data at another location.
    b=load_bundle(root,12,2016,2019)
    assert b.dynamic.ndim==3 and b.dynamic.shape[1]==12
    assert b.train_mask.any() and b.val_mask.any() and b.test_mask.any()
    assert set(b.years[b.train_mask]).isdisjoint(set(b.years[b.test_mask]))
