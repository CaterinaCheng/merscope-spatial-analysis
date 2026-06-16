"""
371_CD8_PVS_DEG.py
DEGs of CD8 T cells in the perivascular space (PVS, <=30um to vessel) vs the rest
(vessel-adjacent + parenchymal). Pool CD8 TRM1/TRM2/TEMRA. Wilcoxon on decontam log;
cross-lineage spillover flagged (perivascular T cells carry vascular/microglial ambient).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import warnings; warnings.filterwarnings("ignore"); sc.settings.verbosity=0
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans"})
NEW=Path(r"<MERSCOPE_ROOT>\merged_analysis\scHPF\new"); QC=Path(r"<MERSCOPE_ROOT>\QC data")
DEC=Path(r"<MERSCOPE_ROOT>\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
VESSEL=["End","Per","SMC"]; CD8=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA"]; NUP,NDN=12,12
UP_RED="#C0392B"; DN_BLUE="#2471A3"
SPILL=set(("PECAM1 CLDN5 VWF PDGFRB ACTA2 RGS5 NOTCH3 COL1A1 COL3A1 COL4A3 COL9A3 FN1 A2M FLT1 DCN IFITM3 RGS5 "
           "AQP4 GJA1 SLC1A3 SLC1A2 GLUL GFAP AQP9 VCAN PLPP3 SPOCK2 "
           "MOG MAL PLP1 MOBP CNP MBP UGT8 OLIG2 CD22 SORT1 "
           "RBFOX3 SYT1 SNAP25 GAD1 MEG3 NRGN SLC17A7 NELL2 BCL11B "
           "CX3CR1 CSF1R C1QA C1QB C1QC AIF1 TYROBP P2RY12 TMEM119 CTSS CD74 HLA-DRA HLA-DPA1 TREM2 GPNMB CD163 MRC1 FCER1G CYBB").split())
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
is_ves=np.isin(v2,VESSEL); run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object); labv=lab.reindex(idx).values
cd8mask=np.isin(labv,CD8)
ent={}
for d in sorted([x for x in QC.iterdir() if x.is_dir()]):
    csv=d/"cell_metadata.csv"
    if not csv.exists(): continue
    df=pd.read_csv(csv,usecols=["EntityID","center_x","center_y"])
    for eid,x,y in zip(df["EntityID"].astype(str),df["center_x"],df["center_y"]): ent[(d.name,eid)]=(x,y)
mx=np.full(len(idx),np.nan); my=np.full(len(idx),np.nan)
for i,c in enumerate(idx):
    if "_" in c: pre,eid=c.rsplit("_",1); mx[i],my[i]=ent.get((pre,eid),(np.nan,np.nan))
hasxy=np.isfinite(mx); ci=np.where(cd8mask&hasxy)[0]; dV=np.full(len(ci),np.inf)
for r in np.unique(run[ci]):
    sel=np.where(run[ci]==r)[0]; vs=np.where(is_ves&(run==r)&hasxy)[0]
    if len(vs): dd,_=cKDTree(np.column_stack([mx[vs],my[vs]])).query(np.column_stack([mx[ci[sel]],my[ci[sel]]]),k=1); dV[sel]=dd
comp=np.where(dV<=30,"PVS","non-PVS")
print("CD8 PVS:",int((comp=="PVS").sum()),"non-PVS:",int((comp=="non-PVS").sum()))
A=ad.AnnData(X=Xd[ci].copy(),obs=pd.DataFrame({"comp":comp},index=idx[ci]),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
sc.tl.rank_genes_groups(A,"comp",groups=["PVS"],reference="non-PVS",method="wilcoxon")
r=sc.get.rank_genes_groups_df(A,group="PVS"); r["spill"]=r.names.isin(SPILL)
up=r[(r.pvals_adj<0.05)&(r.logfoldchanges>0)].sort_values("logfoldchanges",ascending=False).head(NUP)
dn=r[(r.pvals_adj<0.05)&(r.logfoldchanges<0)].sort_values("logfoldchanges").head(NDN)
DEG=pd.concat([up,dn]); DEG.to_csv(NEW/"CD8_PVS_DEG.csv",index=False)
print("\n=== CD8 in PVS vs non-PVS (intrinsic, [S]=spillover) ===")
print("UP in PVS:",", ".join(f"{g}{'[S]' if s else ''}" for g,s in zip(up.names,up.spill)))
print("DOWN in PVS:",", ".join(f"{g}{'[S]' if s else ''}" for g,s in zip(dn.names,dn.spill)))
# figure
fig,ax=plt.subplots(figsize=(7,6.5)); db=DEG.sort_values("logfoldchanges"); y=np.arange(len(db))
for yi,(_,row) in zip(y,db.iterrows()):
    col=UP_RED if row.logfoldchanges>0 else DN_BLUE
    ax.barh(yi,row.logfoldchanges,color=col,edgecolor="k" if row.spill else "none",lw=0.6,hatch="///" if row.spill else None)
    ha="left" if row.logfoldchanges>0 else "right"; off=0.04 if row.logfoldchanges>0 else -0.04
    ax.text(row.logfoldchanges+off,yi,row.names+(" (amb)" if row.spill else ""),va="center",ha=ha,fontsize=7.5)
ax.axvline(0,color="k",lw=0.9); ax.set_yticks([]); ax.set_xlabel("log2 fold-change (PVS vs non-PVS)")
mx2=np.abs(db.logfoldchanges).max()*1.5; ax.set_xlim(-mx2,mx2)
ax.set_title(f"CD8 T-cell DEGs in perivascular space\n(PVS n={int((comp=='PVS').sum())} vs non-PVS n={int((comp=='non-PVS').sum())}; padj<0.05; hatched=ambient)",fontsize=10,fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=UP_RED,label="up in PVS"),Patch(color=DN_BLUE,label="down in PVS")],fontsize=8,loc="lower right")
for sp in ax.spines.values(): sp.set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"CD8_PVS_DEG.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: CD8_PVS_DEG.png + CD8_PVS_DEG.csv")
