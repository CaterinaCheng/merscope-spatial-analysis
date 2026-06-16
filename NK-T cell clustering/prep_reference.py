"""
292a_prep_reference.py  (Stage 2 prep)
Load abl5197 T-cell raw-counts atlas, subset to the 494 shared panel genes (panel order),
attach curated labels + donor/organ, save a compact reference h5ad for scHPF consensus training.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, anndata as ad
from scipy.sparse import csr_matrix
REF=Path(r"D:\Caterina\MERSCOPE\reference\CountAdded_PIP_T_object_for_cellxgene.h5ad")
PANELH5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad")
OUT=Path(r"D:\Caterina\MERSCOPE\reference\abl5197_T_panel494.h5ad")

with h5py.File(PANELH5,"r") as g:
    panel=[s.decode() if isinstance(s,bytes) else s for s in g["var"][g["var"].attrs.get("_index","_index")][:]]
panel=[p for p in panel if not p.startswith("Blank")]

print("reading reference X (sparse CSR)...")
f=h5py.File(REF,"r")
var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
Xg=f["layers"]["counts"]   # RAW integer counts (X is log-normalized; scHPF needs counts)
X=csr_matrix((Xg["data"][:],Xg["indices"][:],Xg["indptr"][:]),shape=tuple(int(s) for s in Xg.attrs["shape"]))
print("  counts",X.shape,"nnz",X.nnz)
print("  max",X.data.max(),"all-int",np.all(X.data==np.round(X.data)))
vp={g:i for i,g in enumerate(var)}
shared=[p for p in panel if p in vp]; cols=[vp[p] for p in shared]
Xs=X[:,cols].tocsr(); print(f"  subset to {len(shared)} shared genes -> {Xs.shape}")
def col(name):
    g=f["obs"][name]
    if isinstance(g,h5py.Group):
        cats=[s.decode() if isinstance(s,bytes) else s for s in g["categories"][:]]; codes=g["codes"][:]
    else:
        codes=g[:]; cats=[s.decode() if isinstance(s,bytes) else s for s in f["obs"]["__categories"][name][:]]
    return np.array([cats[c] if c>=0 else "NA" for c in codes])
obs=pd.DataFrame({"celltype":col("Manually_curated_celltype"),"donor":col("Donor"),"organ":col("Organ")})
obsn=[s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]]
obs.index=obsn; f.close()
A=ad.AnnData(X=Xs,obs=obs,var=pd.DataFrame(index=shared))
A.write(OUT)
print(f"Saved compact reference: {OUT}  ({A.n_obs} cells x {A.n_vars} genes)")
print("celltypes:",A.obs.celltype.value_counts().to_dict())
