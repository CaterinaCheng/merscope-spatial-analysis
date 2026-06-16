"""
290_allcell_rebuild.py  (Stage 1: all-brain-cell rebuild from raw counts, guide-faithful)
Raw counts (layers/counts) -> light MERSCOPE QC -> normalize+log1p -> VOLUME regression
(guide's SCTransform vars.to.regress='volume') -> scale -> PCA -> Harmony(batch=section)
-> neighbors -> Leiden. Saves embedding + leiden for annotation (Stage 1b).
Starts from RAW counts and IGNORES the old cell_type_v2 labels (rebuild before clustering).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd
import h5py, anndata as ad, scanpy as sc
from scipy.sparse import csr_matrix
sc.settings.verbosity=1; sc.settings.n_jobs=8
CMAP=Path(r"<MERSCOPE_ROOT>\merged_analysis\cellmap"); SAVE=Path(r"<MERSCOPE_ROOT>\merged_analysis\scHPF")
H5=CMAP/"merged_qc_brain_remapped.h5ad"; QC=Path(r"<MERSCOPE_ROOT>\QC data")
OUT=CMAP/"allcell_rebuild.h5ad"

print("loading raw counts + obs...")
with h5py.File(H5,"r") as f:
    og=f["obs"]; idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in og[og.attrs.get("_index","_index")][:]])
    vg=f["var"]; var=[s.decode() if isinstance(s,bytes) else s for s in vg[vg.attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    def cat(name):
        n=og[name]; c=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; return np.array([c[i] for i in n["codes"][:]])
    donor=cat("donor")
run=np.array([cid.rsplit("_",1)[0] if "_" in cid else "?" for cid in idx],dtype=object)

# volume per cell (guide regresses this out); fill missing with global median
print("loading volume...")
vol_map={}
for d in sorted([x for x in QC.iterdir() if x.is_dir()]):
    csv=d/"cell_metadata.csv"
    if not csv.exists(): continue
    head=pd.read_csv(csv,nrows=0).columns
    if "volume" not in head: continue
    df=pd.read_csv(csv,usecols=["EntityID","volume"])
    for eid,v in zip(df["EntityID"].astype(str),df["volume"]): vol_map[(d.name,eid)]=v
vol=np.full(len(idx),np.nan)
for i,cid in enumerate(idx):
    if "_" in cid:
        pre,eid=cid.rsplit("_",1); vol[i]=vol_map.get((pre,eid),np.nan)
vol=np.where(np.isfinite(vol),vol,np.nanmedian(vol)); logvol=np.log1p(vol)
print(f"  volume present for {np.isfinite(vol).mean()*100:.0f}% cells")

A=ad.AnnData(X=X,obs=pd.DataFrame({"donor":donor,"section":run,"logvol":logvol},index=idx),var=pd.DataFrame(index=var))
A.layers["counts"]=A.X.copy()
tot=np.asarray(A.X.sum(1)).ravel()
print(f"total cells={A.n_obs}; median transcripts/cell={np.median(tot):.0f}")
# light MERSCOPE QC (panel is 550 genes; counts are low — keep cells with >=20 transcripts & >=5 genes)
ng=np.asarray((A.X>0).sum(1)).ravel()
keep=(tot>=20)&(ng>=5); print(f"QC keep {keep.sum()}/{len(keep)} ({100*keep.mean():.1f}%) [>=20 transcripts & >=5 genes]")
A=A[keep].copy()

sc.pp.normalize_total(A,target_sum=1e4); sc.pp.log1p(A)
# --- volume regression (guide: SCTransform vars.to.regress='volume'); exact vectorised OLS residualisation ---
print("regressing out volume (vectorised)...")
Y=A.X.toarray() if hasattr(A.X,"toarray") else np.asarray(A.X)
Z=np.column_stack([np.ones(A.n_obs),(A.obs["logvol"].values-A.obs["logvol"].values.mean())])
B=np.linalg.lstsq(Z,Y,rcond=None)[0]; Y=Y-Z@B
A.X=Y.astype(np.float32)
sc.pp.scale(A,max_value=10)
print("PCA..."); sc.tl.pca(A,n_comps=50)
print("Harmony (batch=section)..."); sc.external.pp.harmony_integrate(A,key="section",max_iter_harmony=20)
print("neighbors + Leiden..."); sc.pp.neighbors(A,n_neighbors=30,use_rep="X_pca_harmony")
sc.tl.leiden(A,resolution=1.0,flavor="igraph",n_iterations=2,directed=False)
print(f"Leiden clusters: {A.obs['leiden'].nunique()}")
A.X=A.layers["counts"]  # restore raw counts for saving
A.write(OUT)
A.obs[["donor","section","leiden"]].to_csv(SAVE/"allcell_rebuild_leiden.csv")
print(f"Saved: {OUT} + allcell_rebuild_leiden.csv")
