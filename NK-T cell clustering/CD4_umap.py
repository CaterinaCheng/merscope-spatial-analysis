"""
308_CD4_umap.py
CD4-only UMAP, embedded on the CD4 scHPF intrinsic factor scores (per-lineage representation).
Main panel coloured by final CD4 subset; marker overlays (FOXP3, GZMB/PRF1, IL7R/CCR7, CD40LG)
to judge whether subsets separate.
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
INTR=["F0","F2","F3","F5","F6","F7"]   # CD4 intrinsic (spillover F1,F4 dropped)

lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
cd4lab=lab[lab.isin(["CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg"])]
sc8=pd.read_csv(NEW/"Tcell_CD4_schpf_scores.csv",index_col=0)
d=sc8.join(cd4lab.rename("subset"),how="inner")
print("CD4 cells with scHPF scores:",d.subset.value_counts().to_dict())

A=ad.AnnData(X=np.zeros((len(d),1)),obs=d[["subset"]].copy()); A.obsm["X_f"]=d[INTR].values
sc.pp.neighbors(A,use_rep="X_f",n_neighbors=15); sc.tl.umap(A,min_dist=0.4,random_state=0)
U=A.obsm["X_umap"]; d["u1"],d["u2"]=U[:,0],U[:,1]

# expression for marker overlays
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
vp={gn:i for i,gn in enumerate(var)}; pos={c:i for i,c in enumerate(idx)}
import scanpy as sc2
Ax=ad.AnnData(X=X[[pos[c] for c in d.index]],var=pd.DataFrame(index=var)); sc.pp.normalize_total(Ax,target_sum=1e4); sc.pp.log1p(Ax)
def expr(gn): return np.asarray(Ax[:,gn].X.todense()).ravel() if gn in vp else np.zeros(len(d))

pal={"CD4 Th":"#3498DB","CD4 CTL":"#E74C3C","CD4 Tcm/mem":"#2ECC71","CD4 Treg":"#9B59B6"}
fig,axes=plt.subplots(2,3,figsize=(15,9)); axes=axes.ravel()
# panel 0: subset
for s in [x for x in pal if x in set(d.subset)]:
    m=(d.subset==s).values; axes[0].scatter(U[m,0],U[m,1],s=26,c=pal[s],label=f"{s} ({m.sum()})",linewidths=0.2,edgecolors="white",alpha=0.9)
axes[0].set_title(f"CD4 T cells (n={len(d)}) — scHPF UMAP",fontsize=11,fontweight="bold"); axes[0].legend(markerscale=1.6,fontsize=8)
# marker overlays
for ax,gn in zip(axes[1:],["FOXP3","CTLA4","GZMB","IL7R","CD40LG"]):
    v=expr(gn); sca=ax.scatter(U[:,0],U[:,1],s=22,c=v,cmap="viridis",linewidths=0,alpha=0.9)
    ax.set_title(gn,fontsize=11,fontweight="bold"); fig.colorbar(sca,ax=ax,shrink=0.7)
for ax in axes: ax.set_xticks([]); ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ("top","right")]
plt.tight_layout(); fig.savefig(NEW/"umap_CD4_only.png",dpi=140,bbox_inches="tight"); plt.close()
d[["u1","u2","subset"]].to_csv(NEW/"umap_CD4_only_coords.csv")
print("Saved: umap_CD4_only.png + umap_CD4_only_coords.csv")
