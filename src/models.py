from __future__ import annotations
import math
import torch
from torch import nn

class LSTMRegressor(nn.Module):
    def __init__(self, dyn_dim, soil_dim, hidden=96, dropout=.15):
        super().__init__(); self.rnn=nn.LSTM(dyn_dim,hidden,batch_first=True,bidirectional=True)
        self.head=nn.Sequential(nn.Linear(hidden*2+soil_dim,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))
    def forward(self,x,soil,knowledge=None): return self.head(torch.cat([self.rnn(x)[0].mean(1),soil],1)).squeeze(1)

class TransformerRegressor(nn.Module):
    def __init__(self,dyn_dim,soil_dim,hidden=96,heads=4,dropout=.15):
        super().__init__(); self.inp=nn.Linear(dyn_dim,hidden); e=nn.TransformerEncoderLayer(hidden,heads,hidden*3,dropout,batch_first=True,norm_first=True); self.enc=nn.TransformerEncoder(e,2)
        self.head=nn.Sequential(nn.Linear(hidden+soil_dim,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,1))
    def forward(self,x,soil,knowledge=None): return self.head(torch.cat([self.enc(self.inp(x)).mean(1),soil],1)).squeeze(1)

class LagCrossAttention(nn.Module):
    """Weather queries remote-sensing states with a differentiable physiological lag prior."""
    def __init__(self, weather_dim, remote_dim, hidden, heads, lag_min, lag_max):
        super().__init__(); self.q=nn.Linear(weather_dim,hidden); self.k=nn.Linear(remote_dim,hidden); self.v=nn.Linear(remote_dim,hidden)
        self.out=nn.Linear(hidden,hidden); self.heads=heads; self.d=hidden//heads; assert hidden%heads==0
        self.raw_tau=nn.Parameter(torch.zeros(heads)); self.raw_sigma=nn.Parameter(torch.zeros(heads)); self.lag_min,self.lag_max=lag_min,lag_max
    def forward(self,w,r,knowledge=None,enable_lag=True):
        b,t,_=w.shape; q=self.q(w); k=self.k(r); v=self.v(r)
        if knowledge is not None: q=q+knowledge[:,None,:]
        q,k,v=[z.view(b,t,self.heads,self.d).transpose(1,2) for z in (q,k,v)]
        score=(q@k.transpose(-1,-2))/math.sqrt(self.d)
        pos=torch.arange(t,device=w.device,dtype=w.dtype); qpos=pos[None]-pos[:,None] # key - query
        tau=self.lag_min+(self.lag_max-self.lag_min)*torch.sigmoid(self.raw_tau)
        sigma=.5+2.5*torch.sigmoid(self.raw_sigma)
        if enable_lag: score=score-0.5*((qpos[None,:, :]+tau[:,None,None])/sigma[:,None,None]).square()
        # Causal mask forbids an observed response from preceding its driving weather.
        score=score.masked_fill(qpos[None,None,:,:] > 0, float("-inf"))
        a=torch.softmax(score,dim=-1); z=(a@v).transpose(1,2).reshape(b,t,-1)
        return self.out(z),a,tau

class ProposedModel(nn.Module):
    def __init__(self,dyn_dim,soil_dim,weather_idx,remote_idx,knowledge_dim=0,hidden=96,heads=4,dropout=.15,lag_min=.5,lag_max=3.):
        super().__init__(); self.weather_idx,self.remote_idx=weather_idx,remote_idx; self.knowledge_dim=knowledge_dim
        self.know=nn.Linear(knowledge_dim,hidden) if knowledge_dim else None
        self.cross=LagCrossAttention(len(weather_idx),len(remote_idx),hidden,heads,lag_min,lag_max)
        e=nn.TransformerEncoderLayer(hidden,heads,hidden*3,dropout,batch_first=True,norm_first=True); self.temporal=nn.TransformerEncoder(e,2)
        self.head=nn.Sequential(nn.Linear(hidden+soil_dim,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,1))
        self.last_attention=self.last_tau=None
    def forward(self,x,soil,knowledge=None,enable_lag=True):
        w=x[:,:,self.weather_idx]; r=x[:,:,self.remote_idx]
        k=self.know(knowledge) if self.know is not None and knowledge is not None else None
        z,a,tau=self.cross(w,r,k,enable_lag); z=self.temporal(z); pooled=z.mean(1)
        self.last_attention=a.detach(); self.last_tau=tau.detach()
        return self.head(torch.cat([pooled,soil],1)).squeeze(1)
