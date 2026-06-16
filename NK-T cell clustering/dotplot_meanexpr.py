"""
310_dotplot_meanexpr.py
Subset dot plot with MEAN NORMALIZED EXPRESSION (not z-scored / standard_scale).
Normalization = counts-per-median + log1p (MERSCOPE-appropriate). Adds NK-specific receptor
block (KLRD1/KLRF1/KLRC1/NCR3/FCGR3A) + EOMES per literature on NK-vs-CD8TRM.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
sc.settings.verbosity=0
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
H5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad")
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
vp={gn:i for i,gn in enumerate(var)}; vs=set(var); pos={c:i for i,c in enumerate(idx)}
cells=[c for c in lab.index if c in pos]
A=ad.AnnData(X=X[[pos[c] for c in cells]],obs=pd.DataFrame({"subset":lab.loc[cells].values},index=cells),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)   # counts-per-median + log1p
GROUPS={"Lineage":["CD3D","CD3E","CD3G","CD8A","CD8B","CD4"],
 "Residency/CD103":["CD69","ITGAE","CXCR6","ITGA1","ZNF683"],
 "Memory":["IL7R","CD27","CD28","TCF7","SELL","CCR7"],
 "Cytotoxic":["GZMK","GZMB","GZMA","PRF1","GNLY","NKG7","TBX21","EOMES"],
 "NK-specific":["KLRD1","KLRF1","KLRC1","NCR3","FCGR3A"],
 "Treg":["FOXP3","IL2RA","CTLA4","IKZF2"],"TEMRA/circ":["CX3CR1","S1PR5","FGFBP2","KLRG1"]}
GROUPS={k:[g for g in v if g in vs] for k,v in GROUPS.items()}; GROUPS={k:v for k,v in GROUPS.items() if v}
order=[s for s in ["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"] if s in set(A.obs["subset"])]
A.obs["subset"]=pd.Categorical(A.obs["subset"],categories=order,ordered=True)
sc.pl.dotplot(A,GROUPS,groupby="subset",standard_scale=None,show=False,figsize=(16,4.6),dot_max=0.85,
              cmap="Reds",colorbar_title="mean expr\n(log1p, per-median)")
plt.savefig(NEW/"umap_Tcell_final_dotplot.png",dpi=140,bbox_inches="tight"); plt.close()
print("Saved: umap_Tcell_final_dotplot.png (mean normalized expression)")
