"""
291b_finalize_annotation.py  (Stage 1b final, per user decisions)
- confident clusters: data-driven lineage label
- low-count/ambiguous clusters: retain previous cell_type_v2 (old_v2) per cell
- myeloid clusters {6,12,19}: subcluster + marker-resolve Microglia vs Mono/Mac (BAM/peripheral)
Saves cell_type_rebuild (final all-cell taxonomy).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc
sc.settings.verbosity=0
CMAP=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap"); SAVE=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF")
A=sc.read_h5ad(CMAP/"allcell_rebuild.h5ad")
A.X=A.layers["counts"].copy(); sc.pp.normalize_total(A,target_sum=1e4); sc.pp.log1p(A)
vp=set(A.var_names)
CONF={"0":"End","2":"Oli","4":"Exc","5":"Per","7":"Ast","9":"Oli","10":"Oli","11":"OPC","13":"T/NK","16":"Oli"}
MYELOID={"6","12","19"}; AMBIG={"1","3","8","14","15","17","18","20"}
lei=A.obs.leiden.astype(str).values; old=A.obs["old_v2"].astype(str).values
ct=np.array(["?"]*A.n_obs,dtype=object)
for cl,lab in CONF.items(): ct[lei==cl]=lab
for cl in AMBIG: ct[lei==cl]=old[lei==cl]   # retain previous labels for low-count/ambiguous

# --- myeloid subcluster + marker resolve ---
mmask=np.isin(lei,list(MYELOID)); M=A[mmask].copy()
sc.pp.neighbors(M,n_neighbors=20,use_rep="X_pca_harmony"); sc.tl.leiden(M,resolution=0.6,flavor="igraph",n_iterations=2,directed=False)
mic=[g for g in ["CX3CR1","TREM2","C1QA","CSF1R","P2RY12","TMEM119"] if g in vp]
mac=[g for g in ["CD163","MRC1","F13A1","COLEC12","SIGLEC1","VSIG4"] if g in vp]
per=[g for g in ["FPR1","AQP9","SERPINA1","S100A8","S100A9","LYZ","SLC11A1"] if g in vp]
sc.tl.score_genes(M,mic,score_name="mic"); sc.tl.score_genes(M,mac,score_name="mac"); sc.tl.score_genes(M,per,score_name="per")
sub=M.obs.groupby("leiden")[["mic","mac","per"]].mean()
print("myeloid subclusters (mean scores):"); print(sub.round(3).to_string())
subassign={}
for cl,r in sub.iterrows():
    subassign[cl]= "Mono/Mac" if (r["mac"]>=r["mic"] or r["per"]>r["mic"]) else "Mic"
print("myeloid subcluster -> assignment:",subassign)
mlabel=M.obs.leiden.map(subassign).astype(str).values
ct[mmask]=mlabel
A.obs["cell_type_rebuild"]=ct
A.obs[["leiden","cell_type_rebuild","old_v2"]].to_csv(SAVE/"allcell_rebuild_celltypes_final.csv")
A.write(CMAP/"allcell_rebuild.h5ad")

print("\nFINAL cell_type_rebuild counts:"); print(pd.Series(ct).value_counts().to_string())
print("\nold_v2 counts (for comparison):"); print(pd.Series(old).value_counts().to_string())
print("\nMicroglia: rebuild=%d (old=%d) ; T/NK: rebuild=%d (old=%d)"%(
    (ct=="Mic").sum(),(old=="Mic").sum(),(ct=="T/NK").sum(),(old=="T/NK").sum()))
print("\nSaved: allcell_rebuild_celltypes_final.csv (+ updated h5ad)")
