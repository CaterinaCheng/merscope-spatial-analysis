"""
307_treg_qc.py
QC the rescued Tregs: real Tregs should NOT express PRF1 (perforin) or TBX21 (T-bet).
Check PRF1/TBX21/GZMB in Treg vs other subsets; flag contaminants; offer a clean rescue
(FOXP3+corroborated AND PRF1- AND TBX21-).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py
from scipy.sparse import csr_matrix
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
H5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad")
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"]))
vp={gn:i for i,gn in enumerate(var)}; pos={c:i for i,c in enumerate(idx)}
def E(gn): return np.asarray(X[:,vp[gn]].todense()).ravel() if gn in vp else np.zeros(X.shape[0])
GENES=["PRF1","TBX21","GZMB","NKG7","GNLY","FOXP3","IL2RA","CTLA4","IKZF2"]
expr={g:E(g) for g in GENES if g in vp}

print("% expressing (count>0) per subset:")
hdr="subset".ljust(16)+"".join(g.rjust(8) for g in expr)
print(hdr+"   n")
for s in ["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"]:
    ci=np.array([pos[c] for c in lab[lab==s].index if c in pos])
    if len(ci)==0: continue
    row=s.ljust(16)+"".join(f"{100*(expr[g][ci]>0).mean():7.0f}%" for g in expr)+f"  {len(ci)}"
    print(row)

# Treg-specific: how many rescued Tregs are PRF1+ or TBX21+
tr=np.array([pos[c] for c in lab[lab=="CD4 Treg"].index if c in pos])
prf=expr.get("PRF1",np.zeros(X.shape[0]))[tr]; tbx=expr.get("TBX21",np.zeros(X.shape[0]))[tr]
contam=(prf>0)|(tbx>0)
print(f"\nCD4 Treg (n={len(tr)}): PRF1+={int((prf>0).sum())}  TBX21+={int((tbx>0).sum())}  PRF1+ or TBX21+ = {int(contam.sum())} ({100*contam.mean():.0f}%)")
print("  -> these are likely cytotoxic/Th1 contaminants that slipped in via CTLA4/TIGIT corroboration")
print(f"  clean Tregs (FOXP3-signature, PRF1- AND TBX21-): {int((~contam).sum())}")

# write a cleaned label: demote contaminated Tregs
clean=lab.copy()
treg_ids=lab[lab=="CD4 Treg"].index
for c,bad in zip(treg_ids,contam):
    if bad: clean.loc[c]="CD4 T (cytotoxic/Th1 - not Treg)"
clean.name="subset_final"
print(f"\nTreg after removing PRF1+/TBX21+ contaminants: {int((clean=='CD4 Treg').sum())}")
clean.to_csv(NEW/"Tcell_subset_final_labels_tregQC.csv")
print("Saved: Tcell_subset_final_labels_tregQC.csv (clean Treg set)")
