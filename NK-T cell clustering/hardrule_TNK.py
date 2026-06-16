"""
297_hardrule_TNK.py
Enforce hard biological rules on the T/NK compartment instead of trusting label-transfer
fine subtypes (MAIT/ILC3/Tgd/NK-subsets are NOT definable on this panel).
 Rules:  T cell  = CD3 (CD3D/E/G) expressed.
         NK cell = CD3-negative AND CD8B-negative AND NK markers expressed.
Check marker availability (esp. MAIT-defining), classify, and show how over-claimed the
transferred labels are (e.g. fraction of 'NK'-transferred cells that are actually CD3+ T).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new"); LAB=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF")
H5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad")
from scipy.sparse import csr_matrix

CHECK={"T-defining":["CD3D","CD3E","CD3G","CD247"],"CD8":["CD8A","CD8B"],"CD4":["CD4"],
 "NK":["NKG7","GNLY","KLRD1","KLRF1","KLRC1","FCGR3A","NCR1","NCR3","NCAM1"],
 "MAIT":["SLC4A10","KLRB1","TRAV1-2","ZBTB16","RORC"],"ILC":["KIT","IL7R","GATA3","IL1R1","RORA"],
 "gdT":["TRDC","TRGC1","TRGC2","TRDV1"]}
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"]))
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
vp={gn:i for i,gn in enumerate(var)}; vs=set(var)
print("MARKER AVAILABILITY ON PANEL:")
for grp,gs in CHECK.items():
    print(f"  {grp:11}: present={[g for g in gs if g in vs]}  absent={[g for g in gs if g not in vs]}")

def expr(gene): return np.asarray(X[:,vp[gene]].todense()).ravel() if gene in vp else np.zeros(X.shape[0])
ist=np.where(v2=="T/NK")[0]
cd3=expr("CD3D")[ist]+expr("CD3E")[ist]+expr("CD3G")[ist]
cd8b=expr("CD8B")[ist]; cd8a=expr("CD8A")[ist]; cd4=expr("CD4")[ist]
nk=sum(expr(g)[ist] for g in ["NKG7","GNLY","KLRD1","KLRF1","FCGR3A"])
isT=cd3>=1
isNK=(cd3==0)&(cd8b==0)&(nk>=1)
amb=~(isT|isNK)
print(f"\nT/NK compartment n={len(ist)}")
print(f"  T  (CD3+)                : {isT.sum()} ({100*isT.mean():.0f}%)")
print(f"  NK (CD3- CD8B- NKmarker+) : {isNK.sum()} ({100*isNK.mean():.0f}%)")
print(f"  ambiguous (CD3- but CD8B+ or no NK marker): {amb.sum()} ({100*amb.mean():.0f}%)")
print(f"  [note CD3 detection is sparse in MERSCOPE -> CD3+ is a lower bound on true T]")
within=pd.Series(np.where(isT,"T",np.where(isNK,"NK","ambiguous")),index=idx[ist])

# how over-claimed are the transferred labels?
tr=pd.read_csv(NEW/"merscope_Tcell_transferred_celltype.csv").set_index("cell_id")["ref_celltype"].reindex(idx[ist])
cx=pd.crosstab(tr.values,within.values)
print("\nTRANSFERRED ref-celltype  x  hard-rule class (counts):")
print(cx.to_string())
print("\n=> fraction of each transferred label that is actually CD3+ T:")
for lab in cx.index:
    tot=cx.loc[lab].sum(); tfrac=cx.loc[lab].get("T",0)/tot
    print(f"   {lab:24} CD3+T={cx.loc[lab].get('T',0):4}/{tot:<4} ({100*tfrac:.0f}%)")

# existing phenotype CD3 check
ph=pd.concat([pd.read_csv(LAB/"schpf_CD8_final_labels.csv")[["cell_id","phenotype"]],
              pd.read_csv(LAB/"schpf_CD4_final_labels.csv")[["cell_id","phenotype"]]]).set_index("cell_id")["phenotype"]
phv=ph.reindex(idx[ist])
print("\nexisting phenotype x hard-rule class:")
print(pd.crosstab(phv.values,within.values).to_string())
pd.DataFrame({"cell_id":idx[ist],"cd3":cd3,"cd8b":cd8b,"nk":nk,"hardrule":within.values,
              "transferred":tr.values,"existing_phenotype":phv.values}).to_csv(NEW/"TNK_hardrule_classification.csv",index=False)
print("\nSaved: TNK_hardrule_classification.csv")
