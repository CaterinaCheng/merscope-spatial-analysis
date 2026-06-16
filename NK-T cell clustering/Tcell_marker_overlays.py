"""
309_Tcell_marker_overlays.py
Marker feature plots on the combined T/NK UMAP. Normalization = counts-per-median + log1p
(MERSCOPE-appropriate; avoids the 1e4-target inflation that made FOXP3 look high).
Markers: GZMB, GZMK, FOXP3, CXCR6, TBX21 (requested) + context (ITGAE, NKG7, IL7R, CD8A, CD4, FCGR3A, PRF1).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
sc.settings.verbosity=0; plt.rcParams.update({"font.size":9})
NEW=Path(r"<MERSCOPE_ROOT>\merged_analysis\scHPF\new")
H5=Path(r"<MERSCOPE_ROOT>\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad")
co=pd.read_csv(NEW/"umap_Tcell_final_coords.csv",index_col=0)
MARKERS=["GZMB","GZMK","FOXP3","CXCR6","TBX21","ITGAE","NKG7","IL7R","CD8A","CD4","FCGR3A","PRF1"]

with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
vp={gn:i for i,gn in enumerate(var)}; pos={c:i for i,c in enumerate(idx)}
cells=[c for c in co.index if c in pos]; co=co.loc[cells]; U=co[["u1","u2"]].values
A=ad.AnnData(X=X[[pos[c] for c in cells]],var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None)  # target=None -> median total counts (MERSCOPE-appropriate)
sc.pp.log1p(A)
M=[m for m in MARKERS if m in vp]
fig,axes=plt.subplots(3,4,figsize=(17,11)); axes=axes.ravel()
for ax,m in zip(axes,M):
    v=np.asarray(A[:,m].X.todense()).ravel(); o=np.argsort(v)  # plot high on top
    sca=ax.scatter(U[o,0],U[o,1],s=11,c=v[o],cmap="viridis",linewidths=0,vmax=np.quantile(v[v>0],0.98) if (v>0).any() else 1)
    pctpos=100*(v>0).mean()
    ax.set_title(f"{m}  ({pctpos:.0f}% +)",fontsize=11,fontweight="bold"); fig.colorbar(sca,ax=ax,shrink=0.7)
    ax.set_xticks([]); ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ("top","right")]
for ax in axes[len(M):]: ax.axis("off")
fig.suptitle("T/NK UMAP — marker overlays (log1p counts-per-median; % = fraction of cells detected)",fontsize=12,fontweight="bold",y=1.0)
plt.tight_layout(); fig.savefig(NEW/"umap_Tcell_markers.png",dpi=140,bbox_inches="tight"); plt.close()
print("Saved: umap_Tcell_markers.png ; markers:",M)
