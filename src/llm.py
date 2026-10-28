"""Offline, cacheable LLM knowledge embeddings. No API call occurs during model training."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from openai import OpenAI
from .data import PROVINCES

def _load_dotenv(path: str = ".env") -> None:
    p=Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k,v=line.split("=",1); os.environ.setdefault(k.strip(),v.strip())

def build_region_prompts(bundle, template_path: str) -> dict[str,str]:
    template=Path(template_path).read_text(encoding="utf-8")
    prompts={}
    for i, adm in enumerate(sorted(set(bundle.regions))):
        mask=(bundle.regions==adm) & bundle.train_mask
        # Statistics are calculated only from training-period predictors.
        x=bundle.dynamic[mask]
        lines=[f"{n}: mean={x[:,:,j].mean():.3f}, sd={x[:,:,j].std():.3f}" for j,n in enumerate(bundle.feature_names)]
        prompts[adm]=template.format(region_name=PROVINCES.get(adm,adm), adm_id=adm,
             latitude=bundle.region_coords[i,0], longitude=bundle.region_coords[i,1], sos=1, eos=365,
             statistics="; ".join(lines))
    return prompts

def cached_knowledge_embeddings(bundle, cache_path: str, model: str | None = None, template_path: str = "prompts/agronomic_knowledge_prompt.txt") -> np.ndarray:
    """Generate a knowledge note then embed it with an OpenAI-compatible API; cache by region."""
    cache=Path(cache_path); cache.parent.mkdir(parents=True, exist_ok=True)
    unique=sorted(set(bundle.regions))
    if cache.exists():
        z=np.load(cache,allow_pickle=False)
        if list(z["regions"]) == unique: return z["embeddings"]
    _load_dotenv()
    key=os.getenv("OPENAI_API_KEY")
    if not key: raise RuntimeError("OPENAI_API_KEY is required for --use-llm; put it in .env, never in code.")
    client=OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL") or None)
    chat_model=model or os.getenv("LLM_MODEL","gpt-4.1-mini")
    prompts=build_region_prompts(bundle,template_path)
    notes=[]
    for adm in unique:
        r=client.chat.completions.create(model=chat_model, temperature=0, messages=[{"role":"user","content":prompts[adm]}])
        notes.append(r.choices[0].message.content or "")
    # Replace with an embedding endpoint available at your institution if text-embedding-3-small is unavailable.
    # er=client.embeddings.create(model=os.getenv("EMBEDDING_MODEL","text-embedding-3-small"), input=notes)
    # 修改为分批请求（每批最多 10 条）
    em_list = []
    batch_size = 10
    for i in range(0, len(notes), batch_size):
        batch_notes = notes[i:i+batch_size]
        er = client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"), 
            input=batch_notes
        )
        # 提取每一批的 embedding
        batch_em = np.asarray([x.embedding for x in er.data], dtype=np.float32)
        em_list.append(batch_em)
    
    # 把所有批次的结果拼接起来
    em = np.concatenate(em_list, axis=0)
    em=np.asarray([x.embedding for x in er.data],dtype=np.float32)
    np.savez_compressed(cache,regions=np.asarray(unique),embeddings=em,notes=np.asarray(notes))
    return em
