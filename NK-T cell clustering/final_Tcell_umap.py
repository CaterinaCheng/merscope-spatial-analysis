"""
304_final_Tcell_umap.py
Finalize T/NK subset labels and make the definitive T-cell UMAP + dot plot.
 - combine the two CD4 memory clusters -> single 'CD4 Tcm/mem'
 - rename CD8 TRM subsets -> 'CD8 TRM 1' (CD103+ memory) / 'CD8 TRM 2' (CD103- GZMK+ effector)
Combined T/NK UMAP embedded on lineage+state genes, coloured by final subset.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
sc.settings.verbosity=0
NEW=Path(r"<MERSCOPE_ROOT>\merged_analysis\scHPF\new")
H5=Path(r"<MERSCOPE_ROOT>\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad")

lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
RENAME={"CD8 TRM (CD103+ memory)":"CD8 TRM 1","CD8 TRM (CD103- GZMK+ effector)":"CD8 TRM 2",
        "CD4 Tcm/mem (broad)":"CD4 Tcm/mem"}   # combine the two memory clusters
lab=lab.replace(RENAME)
lab.name="subset_final"; lab.to_csv(NEW/"Tcell_subset_final_labels.csv")
print("FINAL subsets:",lab.value_counts().to_dict())

with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
pos={c:i for i,c in enumerate(idx)}; vs=set(var)
cells=[c for c in lab.index if c in pos]; A=ad.AnnData(X=X[[pos[c] for c in cells]],obs=pd.DataFrame({"subset":lab.loc[cells].values},index=cells),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=1e4); sc.pp.log1p(A)
LINSTATE=[g for g in ["CD3D","CD3E","CD3G","CD2","CD8A","CD8B","CD4","CD40LG","GZMK","GZMB","GZMA","GZMH","PRF1","GNLY","NKG7","TBX21","EOMES",
 "CD69","ITGAE","CXCR6","ITGA1","ZNF683","IL7R","CD27","CD28","TCF7","LEF1","CCR7","SELL","PDCD1","CTLA4","FOXP3","IL2RA",
 "FCGR3A","CX3CR1","S1PR5","FGFBP2","KLRG1","KLRD1","KLRF1","KLRC1"] if g in vs]
Ae=A[:,LINSTATE].copy(); sc.pp.scale(Ae,max_value=10); sc.tl.pca(Ae,n_comps=20); sc.pp.neighbors(Ae,n_neighbors=15); sc.tl.umap(Ae,min_dist=0.4,random_state=0)
U=Ae.obsm["X_umap"]; A.obsm["X_umap"]=U
pd.DataFrame(U,index=cells,columns=["u1","u2"]).join(lab.rename("subset")).to_csv(NEW/"umap_Tcell_final_coords.csv")

pal={"CD8 TRM 1":"#C0392B","CD8 TRM 2":"#7B241C","CD8 TEMRA":"#E67E22","CD4 Th":"#2E86C1","CD4 CTL":"#7D3C98",
     "CD4 Tcm/mem":"#229954","CD4 Treg":"#000000","NK":"#16A085"}
order=[s for s in pal if s in set(A.obs["subset"])]
fig,ax=plt.subplots(figsize=(8.5,7))
for s in order:
    m=(A.obs["subset"]==s).values; ax.scatter(U[m,0],U[m,1],s=18,c=pal[s],label=f"{s} ({m.sum()})",linewidths=0,alpha=0.85)
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
ax.set_title(f"T/NK subsets (n={A.n_obs}) — final annotation",fontsize=11,fontweight="bold")
ax.legend(markerscale=2,fontsize=9,loc="center left",bbox_to_anchor=(1.0,0.5),frameon=False)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"umap_Tcell_final.png",dpi=140,bbox_inches="tight"); plt.close()

# final dot plot
GROUPS={"Lineage":["CD3D","CD3E","CD8A","CD8B","CD4"],"Residency/CD103":["CD69","ITGAE","CXCR6","ITGA1","ZNF683"],
 "Memory/quiescent":["IL7R","CD27","CD28","TCF7","LEF1","CCR7","SELL","PDCD1","CTLA4"],
 "Cytotoxic/effector":["GZMK","GZMB","GZMA","GZMH","PRF1","GNLY","NKG7","TBX21","EOMES"],
 "Circulating (TEMRA)":["FCGR3A","FGFBP2","CX3CR1","S1PR5","KLRG1"],"Treg":["FOXP3","IL2RA","IKZF2","TIGIT"],"NK":["KLRD1","KLRF1","KLRC1"]}
GROUPS={k:[g for g in v if g in vs] for k,v in GROUPS.items()}; GROUPS={k:v for k,v in GROUPS.items() if v}
A.obs["subset"]=pd.Categorical(A.obs["subset"],categories=order,ordered=True)
sc.pl.dotplot(A,GROUPS,groupby="subset",standard_scale="var",show=False,figsize=(15,4.4),dot_max=0.8,colorbar_title="scaled mean")
plt.savefig(NEW/"umap_Tcell_final_dotplot.png",dpi=140,bbox_inches="tight"); plt.close()
print("Saved: umap_Tcell_final.png + umap_Tcell_final_dotplot.png + umap_Tcell_final_coords.csv")
