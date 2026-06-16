"""
319_compartment_by_celltype.py
Vascular compartment (perivascular <=50um / vessel-adjacent 50-100 / parenchymal >100) for
EVERY cell type + the T subsets. Distance to nearest vessel (End/Per/SMC) per section.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, matplotlib.pyplot as plt
from scipy.spatial import cKDTree
plt.rcParams.update({"font.size":9})
NEW=Path(r"<MERSCOPE_ROOT>\merged_analysis\scHPF\new")
H5=Path(r"<MERSCOPE_ROOT>\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad"); QC=Path(r"<MERSCOPE_ROOT>\QC data")
VESSEL=["End","Per","SMC"]
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
# combined label: T subset where available, else cell_type_v2
clabel=np.array(v2,dtype=object); ls=lab.reindex(idx)
for i,c in enumerate(idx):
    if isinstance(ls.iloc[i],str): clabel[i]=ls.iloc[i]
is_ves=np.isin(v2,VESSEL); run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object)
ent={}
for d in sorted([x for x in QC.iterdir() if x.is_dir()]):
    csv=d/"cell_metadata.csv"
    if not csv.exists(): continue
    df=pd.read_csv(csv,usecols=["EntityID","center_x","center_y"])
    for eid,x,y in zip(df["EntityID"].astype(str),df["center_x"],df["center_y"]): ent[(d.name,eid)]=(x,y)
mx=np.full(len(idx),np.nan); my=np.full(len(idx),np.nan)
for i,c in enumerate(idx):
    if "_" in c: pre,eid=c.rsplit("_",1); mx[i],my[i]=ent.get((pre,eid),(np.nan,np.nan))
hasxy=np.isfinite(mx); dV=np.full(len(idx),np.inf)
for r in np.unique(run):
    cs=np.where((run==r)&hasxy)[0]; vs=np.where(is_ves&(run==r)&hasxy)[0]
    if len(vs):
        dd,_=cKDTree(np.column_stack([mx[vs],my[vs]])).query(np.column_stack([mx[cs],my[cs]]),k=1); dV[cs]=dd
comp=np.where(dV<=30,"perivascular",np.where(dV<=100,"vessel-adjacent","parenchymal"))  # calibrated cutoffs
df=pd.DataFrame({"celltype":clabel,"compartment":comp})[hasxy&np.isfinite(dV)]
order=["End","Per","SMC","Mono/Mac","B","Mic","CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK","Ast","OPC","Oli","Inh","Exc"]
order=[o for o in order if o in set(df.celltype)]
ct=pd.crosstab(df.celltype,df.compartment,normalize="index")[["perivascular","vessel-adjacent","parenchymal"]].reindex(order)*100
sizes=df.celltype.value_counts()
print("Compartment % by cell type (perivascular / vessel-adj / parenchymal):")
print(ct.round(0).astype(int).to_string())
ct.to_csv(NEW/"compartment_by_celltype.csv")
# figure
fig,ax=plt.subplots(figsize=(9,7)); ccol={"perivascular":"#C0392B","vessel-adjacent":"#E67E22","parenchymal":"#2471A3"}
y=np.arange(len(order)); left=np.zeros(len(order))
for c in ["perivascular","vessel-adjacent","parenchymal"]:
    ax.barh(y,ct[c].values,left=left,color=ccol[c],edgecolor="white",lw=0.5,label=c)
    for yi,(v,l) in enumerate(zip(ct[c].values,left)):
        if v>=6: ax.text(l+v/2,yi,f"{v:.0f}",ha="center",va="center",fontsize=7.5,color="white",fontweight="bold")
    left+=ct[c].values
ax.set_yticks(y); ax.set_yticklabels([f"{o} (n={int(sizes[o])})" for o in order],fontsize=8); ax.invert_yaxis()
ax.set_xlabel("% of cell type"); ax.set_xlim(0,100); ax.axhline(2.5,color="#888",lw=1,ls="--"); ax.axhline(13.5,color="#888",lw=1,ls="--")
ax.set_title("Vascular compartment composition by cell type\n(vessels=End/Per/SMC ~100% peri by definition; dashed = T-subset block)",fontsize=10,fontweight="bold")
ax.legend(fontsize=8,loc="lower right")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"compartment_by_celltype.png",dpi=140,bbox_inches="tight"); plt.close()
print("\nSaved: compartment_by_celltype.png + compartment_by_celltype.csv")
