"""
347_microglia_state_separability.py
Diagnose WHY Phagocytic / Activated-DAM / MHC-II-APC microglia don't separate.
 (a) pairwise correlation of the 6 Green signature scores across microglia (collinearity)
 (b) marker-gene overlap (Jaccard) among the 6 Green signatures on our panel
 (c) which scHPF factor each Green state maps to (corr of theta factors vs signature scores)
     -> if phago/DAM/APC map to the SAME factor, they are not separable on this panel.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore"); sc.settings.verbosity=0
NEW=Path(r"<MERSCOPE_ROOT>\merged_analysis\scHPF\new")
DEC=Path(r"<MERSCOPE_ROOT>\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
sig=pd.read_csv(NEW/"green_mic_state_signatures.csv"); SIG={c:[g for g in sig[c].dropna()] for c in sig.columns}
STATES=list(SIG.keys()); SH={s:s.split(" (")[0] for s in STATES}
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
micidx=np.where(v2=="Mic")[0]
A=ad.AnnData(X=Xd[micidx].copy(),obs=pd.DataFrame(index=idx[micidx]),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
for k,gl in SIG.items(): sc.tl.score_genes(A,[x for x in gl if x in A.var_names],score_name=k,ctrl_size=50)
S=A.obs[STATES]
# (a) signature-score correlation
print("=== (a) Green signature-score correlation across microglia (Spearman) ===")
C=S.corr(method="spearman"); C.index=[SH[s] for s in C.index]; C.columns=[SH[s] for s in C.columns]
print(C.round(2).to_string())
# (b) marker overlap
print("\n=== (b) marker-gene overlap (shared genes / Jaccard) among signatures ===")
for i,a in enumerate(STATES):
    for b in STATES[i+1:]:
        ga,gb=set(SIG[a]),set(SIG[b]); inter=ga&gb; jac=len(inter)/len(ga|gb)
        if inter: print(f"  {SH[a]:22} vs {SH[b]:22}: shared={sorted(inter)} (Jaccard {jac:.2f})")
# (c) scHPF factor vs each state
print("\n=== (c) which scHPF factor each Green state maps to (Spearman theta vs score) ===")
theta=pd.read_csv(NEW/"microglia_schpf_cell_scores.csv",index_col=0)
theta=theta.reindex(A.obs_names)
M=np.zeros((len(STATES),theta.shape[1]))
for i,s in enumerate(STATES):
    for j,fc in enumerate(theta.columns): M[i,j]=spearmanr(S[s].values,theta[fc].values).correlation
MM=pd.DataFrame(M,index=[SH[s] for s in STATES],columns=list(theta.columns))
print(MM.round(2).to_string())
print("\ntop factor per state:")
for s in STATES: print(f"  {SH[s]:22}: {MM.loc[SH[s]].idxmax()} (rho={MM.loc[SH[s]].max():.2f})")
# focus trio
trio=["Phagocytic-myeloid (Mic.7)","Activated-DAM (Mic.12)","MHC-II/APC (Mic.9)"]
print("\n=== focus: the non-separating trio ===")
print("score corr:\n",S[trio].corr(method="spearman").round(2).to_string())
print("shared top factor?",{SH[t]:MM.loc[SH[t]].idxmax() for t in trio})
A.obs[STATES].to_csv(NEW/"microglia_green_scores.csv")
print("\nSaved: microglia_green_scores.csv")
