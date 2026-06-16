"""
302_subset_dotplot.py
Dot plot of lineage/program marker genes across the robust CD8 / CD4 / NK subsets.
size = % expressing, color = mean log-norm expression.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad
from scipy.sparse import csr_matrix
sc.settings.verbosity=0
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
H5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad")

# robust labels
cd8=pd.read_csv(NEW/"Tcell_CD8_robust_label.csv",index_col=0)["robust"]
cd4=pd.read_csv(NEW/"Tcell_CD4_robust_label.csv",index_col=0)["robust"]
lin=pd.read_csv(NEW/"Tcell_lineage_assignment.csv").set_index("cell_id")["lineage"]
nk=lin[lin=="NK"]
lab=pd.concat([cd8.replace({"CD8 TEMRA-like":"CD8 TEMRA"}),cd4,pd.Series("NK",index=nk.index)])
lab=lab[~lab.index.duplicated()]
print("subset counts:",lab.value_counts().to_dict())

with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
pos={c:i for i,c in enumerate(idx)}; cells=[c for c in lab.index if c in pos]; rows=[pos[c] for c in cells]
A=ad.AnnData(X=X[rows],obs=pd.DataFrame({"subset":lab.loc[cells].values},index=cells),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=1e4); sc.pp.log1p(A)

GROUPS={"Lineage":["CD3D","CD3E","CD3G","CD2","CD4","CD8A","CD8B"],
 "Cytotoxic":["GZMA","GZMB","GZMH","GZMK","PRF1","GNLY","NKG7","EOMES"],
 "Residency (TRM)":["CD69","ITGAE","ITGA1","CXCR6","ZNF683"],
 "Effector/circ (TEMRA)":["FCGR3A","FGFBP2","CX3CR1","KLRG1","S1PR5","TBX21"],
 "Naive/memory":["CCR7","SELL","TCF7","LEF1","IL7R"],
 "Costim/helper":["CD40LG","CD27","CD28","ICOS"],
 "Treg":["FOXP3","CTLA4","IL2RA","IKZF2","TIGIT"],
 "NK":["KLRD1","KLRF1","KLRC1"]}
vs=set(A.var_names); GROUPS={k:[g for g in v if g in vs] for k,v in GROUPS.items()}; GROUPS={k:v for k,v in GROUPS.items() if v}
order=[s for s in ["CD8 TRM","CD8 cytotoxic (Tem/Trm)","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","CD4 T (unresolved)","NK"] if s in set(A.obs["subset"])]
A.obs["subset"]=pd.Categorical(A.obs["subset"],categories=order,ordered=True)
dp=sc.pl.dotplot(A,GROUPS,groupby="subset",standard_scale="var",show=False,figsize=(16,4.2),
                 colorbar_title="mean expr\n(scaled)",dot_max=0.8)
import matplotlib.pyplot as plt
plt.savefig(NEW/"Tcell_subset_dotplot.png",dpi=140,bbox_inches="tight"); plt.close()
print("Saved: Tcell_subset_dotplot.png")
