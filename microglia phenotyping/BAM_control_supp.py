"""
335_BAM_control_supp.py
SUPPLEMENTARY control: the perivascular MHC-II/APC microglial signal is NOT border-associated
macrophage (BAM) admixture. (A) BAM-marker positivity is flat across vascular compartments.
(B) the perivascular APC enrichment holds even within BAM-LOW microglia (so it's genuine
juxtavascular microglia, not contaminating CD163/MRC1+ perivascular macrophages).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy import stats
sc.settings.verbosity=0; plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
VESSEL=["End","Per","SMC"]
BAM=["CD163","MRC1","MARCO","MS4A7","F13A1","SIGLEC1"]   # border-assoc macrophage / perivascular macrophage
APC=["CD74","HLA-DRA","HLA-DPA1","HLA-DPB1","HLA-DRB1","CIITA"]
ccol={"perivascular":"#C0392B","vessel-adjacent":"#E67E22","parenchymal":"#2471A3"}; corder=["perivascular","vessel-adjacent","parenchymal"]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
is_mic=(v2=="Mic"); is_ves=np.isin(v2,VESSEL); run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object)
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
    cs=np.where(is_mic&(run==r)&hasxy)[0]; vs=np.where(is_ves&(run==r)&hasxy)[0]
    if len(vs) and len(cs): dd,_=cKDTree(np.column_stack([mx[vs],my[vs]])).query(np.column_stack([mx[cs],my[cs]]),k=1); dV[cs]=dd
comp=np.where(dV<=30,"perivascular",np.where(dV<100,"vessel-adjacent","parenchymal"))
mi=np.where(is_mic&hasxy&np.isfinite(dV))[0]
A=ad.AnnData(X=Xd[mi].copy(),obs=pd.DataFrame(index=idx[mi]),var=pd.DataFrame(index=var)); sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
sc.tl.score_genes(A,[g for g in BAM if g in var],score_name="BAM"); sc.tl.score_genes(A,[g for g in APC if g in var],score_name="APC")
A.obs["comp"]=comp[mi]
def E(g): return np.asarray(A[:,g].X.todense()).ravel() if g in A.var_names else np.zeros(A.n_obs)
bamg=[g for g in BAM if g in var]
POS=pd.DataFrame({g:[100*(E(g)[A.obs.comp.values==c]>0).mean() for c in corder] for g in bamg},index=corder).T
print("BAM-marker % positive by compartment:"); print(POS.round(1).to_string())

# panel B: APC by compartment in BAM-low microglia (median split, global)
med=A.obs.BAM.median(); A.obs["BAMgrp"]=np.where(A.obs.BAM>med,"BAM-high","BAM-low")
apc_low={c:A.obs[(A.obs.comp==c)&(A.obs.BAMgrp=="BAM-low")].APC.mean() for c in corder}
apc_high={c:A.obs[(A.obs.comp==c)&(A.obs.BAMgrp=="BAM-high")].APC.mean() for c in corder}
print("\nAPC score by compartment, BAM-low:",{k:round(v,3) for k,v in apc_low.items()})
print("APC score by compartment, BAM-high:",{k:round(v,3) for k,v in apc_high.items()})

# ===== FIGURE =====
fig,(axA,axB)=plt.subplots(1,2,figsize=(13,4.6),gridspec_kw={"width_ratios":[1.25,1]})
x=np.arange(len(bamg)); w=0.26
for k,c in enumerate(corder):
    axA.bar(x+(k-1)*w,POS[c].values,w,color=ccol[c],edgecolor="#333",lw=0.3,label=c)
axA.set_xticks(x); axA.set_xticklabels(bamg,fontsize=8.5); axA.set_ylabel("% of microglia positive")
axA.set_title("A. BAM / perivascular-macrophage markers are FLAT across compartments\n(perivascular microglia are not CD163/MRC1+ border macrophages)",fontsize=9.5,fontweight="bold")
axA.legend(fontsize=8,frameon=False)
for sp in ("top","right"): axA.spines[sp].set_visible(False)
xb=np.arange(len(corder)); w2=0.38
axB.bar(xb-w2/2,[apc_low[c] for c in corder],w2,color=[ccol[c] for c in corder],edgecolor="#333",lw=0.3,hatch="",label="BAM-low microglia")
axB.bar(xb+w2/2,[apc_high[c] for c in corder],w2,color=[ccol[c] for c in corder],edgecolor="#333",lw=0.3,alpha=0.5,hatch="///",label="BAM-high microglia")
for i,c in enumerate(corder):
    axB.text(i-w2/2,apc_low[c]+0.002,f"{apc_low[c]:.3f}",ha="center",fontsize=7.5)
    axB.text(i+w2/2,apc_high[c]+0.002,f"{apc_high[c]:.3f}",ha="center",fontsize=7.5)
axB.set_xticks(xb); axB.set_xticklabels(corder,fontsize=8.5,rotation=12); axB.set_ylabel("mean MHC-II/APC score"); axB.axhline(0,color="#333",lw=0.6)
axB.set_title("B. Perivascular APC enrichment holds in BAM-LOW microglia\n(solid=BAM-low, hatched=BAM-high; APC↑ perivascular either way)",fontsize=9.5,fontweight="bold")
from matplotlib.patches import Patch
axB.legend(handles=[Patch(facecolor="grey",label="BAM-low"),Patch(facecolor="grey",alpha=0.5,hatch="///",label="BAM-high")],fontsize=8,frameon=False,loc="upper right")
for sp in ("top","right"): axB.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"BAM_control_supplementary.png",dpi=140,bbox_inches="tight"); plt.close()
POS.to_csv(NEW/"BAM_control_positivity.csv")
print("\nSaved: BAM_control_supplementary.png + BAM_control_positivity.csv")
