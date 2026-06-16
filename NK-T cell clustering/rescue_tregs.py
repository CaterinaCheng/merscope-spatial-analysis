"""
306_rescue_tregs.py
Rescue Tregs missed by strict hard-gating. Search CD4 + unassigned (CD3+ CD8-negative) T cells
for the Treg signature: FOXP3 corroborated by IL2RA/CTLA4/IKZF2 (FOXP3 alone has ambient/dropout).
Report rescued counts vs ambient baseline; update final labels.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py
from scipy.sparse import csr_matrix
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
H5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad")

lin=pd.read_csv(NEW/"Tcell_lineage_assignment.csv").set_index("cell_id")
final=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"]))
vp={gn:i for i,gn in enumerate(var)}; vs=set(var)
TREG=[g for g in ["FOXP3","IL2RA","CTLA4","IKZF2","TIGIT"] if g in vs]
print("Treg markers on panel:",TREG)
def E(gn): return np.asarray(X[:,vp[gn]].todense()).ravel() if gn in vp else np.zeros(X.shape[0])
pos={c:i for i,c in enumerate(idx)}
# ambient baseline: FOXP3+ rate in non-immune cells
with h5py.File(H5,"r") as f:
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
foxp3=E("FOXP3")
print(f"FOXP3+ ambient baseline (non-immune cells): {100*(foxp3[np.isin(v2,['Exc','Inh','Ast','Oli','OPC'])]>0).mean():.1f}%")

# candidate pool: CD4 + unassigned (CD3+ CD8-negative)
cand=lin[lin.lineage.isin(["CD4","unassigned"])].index
ci=np.array([pos[c] for c in cand if c in pos]); cand=[c for c in cand if c in pos]
fox=E("FOXP3")[ci]; cor=sum(E(g)[ci] for g in ["IL2RA","CTLA4","IKZF2"] if g in vs)
# rescue rule: FOXP3>=1 AND >=1 corroborating Treg marker
resc=(fox>=1)&(cor>=1)
print(f"\ncandidate pool (CD4+unassigned): {len(cand)}")
print(f"  FOXP3+ (>=1): {(fox>=1).sum()}  | FOXP3+ & corroborated (IL2RA/CTLA4/IKZF2): {resc.sum()}")
rescued=pd.Index(cand)[resc]
print(f"  current labels of rescued cells:")
print(final.reindex(rescued).fillna("(unassigned)").value_counts().to_string())

# update final labels: rescued -> CD4 Treg
new=final.copy()
for c in rescued:
    new.loc[c]="CD4 Treg"
# also add rescued unassigned cells that weren't in final
add=[c for c in rescued if c not in final.index]
if add: new=pd.concat([new,pd.Series("CD4 Treg",index=add)])
new=new[~new.index.duplicated()]; new.name="subset_final"
print(f"\nTreg before={int((final=='CD4 Treg').sum())}  ->  after rescue={int((new=='CD4 Treg').sum())}")
new.to_csv(NEW/"Tcell_subset_final_labels.csv")
print("\nUpdated final subset counts:"); print(new.value_counts().to_string())
print("\nSaved updated Tcell_subset_final_labels.csv")
