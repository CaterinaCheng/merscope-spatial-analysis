"""
299_TNK_umap_clean.py
Drop ambiguous cells; re-embed T (CD3+) and NK (CD3- CD8B- NK+) on EXPRESSION (so lineage
markers CD3/CD8/NKG7/GNLY drive the layout) -> cleaner T vs NK separation.
Panel A: hard-rule T vs NK.  Panel B: validated CD3+ T phenotypes + NK.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
sc.settings.verbosity=0; plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
H5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad")
hr=pd.read_csv(NEW/"TNK_hardrule_classification.csv").set_index("cell_id")
keep=hr[hr.hardrule.isin(["T","NK"])].copy()   # drop ambiguous
print("kept T+NK:",keep.hardrule.value_counts().to_dict())

with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
pos={c:i for i,c in enumerate(idx)}; rows=[pos[c] for c in keep.index]
A=ad.AnnData(X=X[rows],obs=keep.copy(),var=pd.DataFrame(index=var))
A=A[:,~A.var_names.str.startswith("Blank")].copy()
sc.pp.normalize_total(A,target_sum=1e4); sc.pp.log1p(A)
# embed on LINEAGE-discriminating genes so T (CD3/CD8) vs NK (KLR/FCGR3A) drive the layout
LINEAGE=["CD3D","CD3E","CD3G","CD247","CD8A","CD8B","CD4","CD2","CD5","CD28","CD27","LCK","ZAP70","THEMIS","IL7R","CD40LG","TCF7","CCR7","SELL",
         "NKG7","GNLY","KLRD1","KLRF1","KLRC1","FCGR3A","KLRB1","NCR3","FGFBP2","CX3CR1","FCER1G","TYROBP","IL2RB","KIR2DL3","S1PR5"]
lg=[g for g in LINEAGE if g in set(A.var_names)]
print(f"embedding on {len(lg)} lineage-discriminating genes")
Ae=A[:,lg].copy(); sc.pp.scale(Ae,max_value=10)
sc.tl.pca(Ae,n_comps=min(20,len(lg)-1)); sc.pp.neighbors(Ae,n_neighbors=15); sc.tl.umap(Ae,min_dist=0.4,spread=1.2,random_state=0)
A.obsm["X_umap"]=Ae.obsm["X_umap"]
sc.pp.scale(A,max_value=10)   # for marker sanity below
U=A.obsm["X_umap"]; A.obs["u1"],A.obs["u2"]=U[:,0],U[:,1]
realph={"CD8 TRM","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg"}
A.obs["panelB"]=[p if p in realph else ("NK" if h=="NK" else "T (CD3+, unphenotyped)")
                 for p,h in zip(A.obs.existing_phenotype,A.obs.hardrule)]

fig,axes=plt.subplots(1,2,figsize=(15,6))
for k,c in {"T":"#2166AC","NK":"#B2182B"}.items():
    m=(A.obs.hardrule==k).values; axes[0].scatter(U[m,0],U[m,1],s=16,c=c,label=f"{k} ({m.sum()})",linewidths=0,alpha=0.8)
axes[0].set_title("T vs NK (expression UMAP, ambiguous removed)\nT=CD3+ · NK=CD3- CD8B- NKmarker+",fontsize=10,fontweight="bold"); axes[0].legend(markerscale=2,fontsize=10)
bp={"CD8 TRM":"#C44E52","CD8 TEMRA":"#E67E22","CD4 Th":"#4C72B0","CD4 CTL":"#8172B3","CD4 Tcm/mem":"#55A868","CD4 Treg":"#000000","T (CD3+, unphenotyped)":"#9ecae1","NK":"#B2182B"}
for k,c in bp.items():
    m=(A.obs.panelB==k).values
    if m.sum(): axes[1].scatter(U[m,0],U[m,1],s=16,c=c,label=f"{k} ({m.sum()})",linewidths=0,alpha=0.85)
axes[1].set_title("validated CD3+ T phenotypes + NK",fontsize=10,fontweight="bold"); axes[1].legend(markerscale=2,fontsize=8)
for ax in axes: ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); [ax.spines[s].set_visible(False) for s in ("top","right")]
fig.suptitle(f"T/NK compartment, ambiguous removed (n={A.n_obs}: {int((A.obs.hardrule=='T').sum())} T + {int((A.obs.hardrule=='NK').sum())} NK)",fontsize=11,fontweight="bold",y=1.02)
plt.tight_layout(); fig.savefig(NEW/"umap_TNK_hardrule_clean.png",dpi=140,bbox_inches="tight"); plt.close()
# quick marker sanity: mean CD3 vs NKG7 per class
for gn in ["CD3E","CD8B","NKG7","GNLY","KLRD1"]:
    if gn in A.var_names:
        v=np.asarray(A[:,gn].X).ravel()
        print(f"  {gn}: T={v[(A.obs.hardrule=='T').values].mean():.2f}  NK={v[(A.obs.hardrule=='NK').values].mean():.2f}")
print("Saved: umap_TNK_hardrule_clean.png")
