"""
350_diagnose_mixed.py
What are the 'Mixed/low' microglia clusters (0,2,11)? Check:
 - sequencing depth / complexity (total counts, n_genes) -> low-quality?
 - dominant scHPF factor per cluster -> which microglial PROGRAM they carry
 - top DE genes (Mixed vs rest) -> what they actually express
 - their RAW (non-z) Green scores -> uniformly low, or a state just below threshold?
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad
from scipy.sparse import csr_matrix
import warnings; warnings.filterwarnings("ignore"); sc.settings.verbosity=0
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
co=pd.read_csv(NEW/"microglia_schpf_5state_coords.csv",index_col=0)
theta=pd.read_csv(NEW/"microglia_schpf_cell_scores.csv",index_col=0)
gscore=pd.read_csv(NEW/"microglia_green_scores.csv",index_col=0)   # 6 raw Green scores
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    raw=f["layers/counts"]; Xr=csr_matrix((raw["data"][:],raw["indices"][:],raw["indptr"][:]),shape=tuple(int(s) for s in raw.attrs["shape"])).astype(np.float32)
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
micidx=np.where(v2=="Mic")[0]; mid=idx[micidx]
tot_raw=np.asarray(Xr[micidx].sum(1)).ravel(); ngenes=np.asarray((Xr[micidx]>0).sum(1)).ravel()
Q=pd.DataFrame({"tot":tot_raw,"ngenes":ngenes},index=mid)
co=co.reindex(mid); theta=theta.reindex(mid); gscore=gscore.reindex(mid)
co["mixed"]=co.state=="Mixed/low"
print("=== depth/complexity: Mixed vs assigned ===")
for grp,mask in [("Mixed",co.mixed.values),("Assigned",~co.mixed.values)]:
    print(f"  {grp:9}: n={mask.sum():6d}  median total counts={np.median(Q.tot[mask]):.0f}  median n_genes={np.median(Q.ngenes[mask]):.0f}")
print("\n=== per Mixed cluster: depth + dominant scHPF factor + top raw Green state ===")
FAC=list(theta.columns)
for cl in sorted(co.leiden.unique()):
    m=(co.leiden.values==cl)
    if not co.mixed[m].any(): continue
    mf=theta[m].mean(); topf=mf.sort_values(ascending=False)
    gm=gscore[m].mean(); topg=gm.sort_values(ascending=False)
    print(f"  cl{cl:>2} n={m.sum():5d} med_counts={np.median(Q.tot[m]):.0f} med_genes={np.median(Q.ngenes[m]):.0f} | top factors: {topf.index[0]}={topf.iloc[0]:.2f},{topf.index[1]}={topf.iloc[1]:.2f} | top Green(raw): {topg.index[0].split(' (')[0]}={topg.iloc[0]:.3f}")
# DE genes: Mixed vs rest (decontam log)
A=ad.AnnData(X=Xd[micidx].copy(),obs=pd.DataFrame({"grp":np.where(co.mixed.values,"Mixed","Assigned")},index=mid),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
sc.tl.rank_genes_groups(A,"grp",groups=["Mixed"],reference="Assigned",method="wilcoxon")
up=[A.uns["rank_genes_groups"]["names"]["Mixed"][i] for i in range(20)]
dn=[A.uns["rank_genes_groups"]["names"]["Mixed"][-(i+1)] for i in range(20)]
print("\n=== Mixed vs Assigned, top UP in Mixed:",", ".join(up))
print("=== top DOWN in Mixed (higher in Assigned):",", ".join(dn))
