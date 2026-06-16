"""
336_microglia_umap.py
Microglia UMAP (decontaminated counts, 83k cells). Colored by:
 (A) dominant Green2024 microglial state (argmax of literature-anchored signature scores)
 (B) vascular compartment (peri<=30 / adj 30-100 / paren>=100)
 (C) 6 Green state signature scores (continuous small multiples)
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
sc.settings.verbosity=0; plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
VESSEL=["End","Per","SMC"]
sig=pd.read_csv(NEW/"green_mic_state_signatures.csv"); SIG={c:[g for g in sig[c].dropna()] for c in sig.columns}
STATEORD=list(SIG.keys())
SCOL={"Homeostatic (Mic.2)":"#3498DB","MHC-II/APC (Mic.9)":"#9B59B6","Activated-DAM (Mic.12)":"#E74C3C",
      "Lipid-DAM (Mic.13)":"#E67E22","Inflammatory/IEG (Mic.15)":"#F1C40F","Phagocytic-myeloid (Mic.7)":"#16A085","Mixed/low":"#BDC3C7"}
ccol={"perivascular":"#C0392B","vessel-adjacent":"#E67E22","parenchymal":"#2471A3"}
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
is_mic=(v2=="Mic"); is_ves=np.isin(v2,VESSEL); micidx=np.where(is_mic)[0]; print("microglia:",len(micidx))
A=ad.AnnData(X=Xd[micidx].copy(),obs=pd.DataFrame(index=idx[micidx]),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A); A.raw=A
det=np.asarray((A.X>0).mean(0)).ravel(); keep=np.array(var)[det>=0.05]
Asc=A[:,keep].copy(); sc.pp.scale(Asc,max_value=10); sc.tl.pca(Asc,n_comps=30); sc.pp.neighbors(Asc,n_neighbors=15,n_pcs=30); sc.tl.umap(Asc)
A.obsm["X_umap"]=Asc.obsm["X_umap"]
for k,genes in SIG.items(): sc.tl.score_genes(A,[g for g in genes if g in A.var_names],score_name=k,ctrl_size=50)
S=A.obs[STATEORD]; Z=(S-S.mean())/S.std()
dom=Z.idxmax(1).values; dom[Z.max(1).values<0.1]="Mixed/low"; A.obs["state"]=dom
# compartment
run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object)
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
comp=np.where(dV<=30,"perivascular",np.where(dV<100,"vessel-adjacent",np.where(np.isfinite(dV),"parenchymal","n/a")))
A.obs["comp"]=comp[micidx]
print("dominant state composition:",{k:round(100*(A.obs.state==k).mean(),1) for k in SCOL if (A.obs.state==k).any()})
U=A.obsm["X_umap"]
pd.DataFrame({"umap1":U[:,0],"umap2":U[:,1],"state":A.obs.state.values,"comp":A.obs.comp.values},index=A.obs_names).to_csv(NEW/"microglia_umap_coords.csv")

# ===== FIGURE =====
fig=plt.figure(figsize=(17,9))
gs=fig.add_gridspec(2,4,height_ratios=[1.35,1])
# A dominant state (big)
axA=fig.add_subplot(gs[0,0:2])
for st in STATEORD+["Mixed/low"]:
    m=A.obs.state.values==st
    if m.sum(): axA.scatter(U[m,0],U[m,1],s=2.5,c=SCOL[st],label=f"{st} ({100*m.mean():.0f}%)",linewidths=0,rasterized=True)
axA.set_title("A. Microglia UMAP — dominant Green2024 state",fontsize=11,fontweight="bold"); axA.set_xticks([]); axA.set_yticks([])
axA.legend(markerscale=4,fontsize=8,loc="upper right",frameon=False)
for sp in axA.spines.values(): sp.set_visible(False)
# B compartment (big)
axB=fig.add_subplot(gs[0,2:4])
for c in ["parenchymal","vessel-adjacent","perivascular"]:
    m=A.obs.comp.values==c
    if m.sum(): axB.scatter(U[m,0],U[m,1],s=2.5,c=ccol[c],label=f"{c} ({100*m.mean():.0f}%)",linewidths=0,rasterized=True)
axB.set_title("B. Microglia UMAP — vascular compartment",fontsize=11,fontweight="bold"); axB.set_xticks([]); axB.set_yticks([])
axB.legend(markerscale=4,fontsize=8,loc="upper right",frameon=False)
for sp in axB.spines.values(): sp.set_visible(False)
# C key marker genes (one per state axis) small multiples
MARK=[("CD74","MHC-II/APC"),("IL1B","Inflammatory/IEG"),("CD68","Phagocytic"),("CX3CR1","Homeostatic")]
def E(g): return np.asarray(A[:,g].X.todense()).ravel() if g in A.var_names else np.zeros(A.n_obs)
for i,(gn,lbl) in enumerate(MARK):
    ax=fig.add_subplot(gs[1,i]); sv=E(gn); vmax=np.percentile(sv,99) or 1
    sca=ax.scatter(U[:,0],U[:,1],s=1.5,c=sv,cmap="viridis",vmin=0,vmax=vmax,linewidths=0,rasterized=True)
    ax.set_title(f"{gn}  ({lbl})",fontsize=8.5,fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sca,ax=ax,shrink=0.55)
    for sp in ax.spines.values(): sp.set_visible(False)
plt.tight_layout()
fig.savefig(NEW/"microglia_umap.png",dpi=150,bbox_inches="tight"); plt.close()
# second figure: all 6 state scores as small multiples
fig2,axes=plt.subplots(2,3,figsize=(15,9))
for ax,st in zip(axes.ravel(),STATEORD):
    sv=A.obs[st].values; vmax=np.percentile(np.abs(sv),98)
    sca=ax.scatter(U[:,0],U[:,1],s=1.5,c=sv,cmap="RdBu_r",vmin=-vmax,vmax=vmax,linewidths=0,rasterized=True)
    ax.set_title(st,fontsize=10,fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
    fig2.colorbar(sca,ax=ax,shrink=0.6,label="score")
    for sp in ax.spines.values(): sp.set_visible(False)
fig2.suptitle("Microglia UMAP — Green2024 state signature scores",fontsize=12,fontweight="bold")
plt.tight_layout(); fig2.savefig(NEW/"microglia_umap_state_scores.png",dpi=150,bbox_inches="tight"); plt.close()
print("Saved: microglia_umap.png + microglia_umap_state_scores.png + microglia_umap_coords.csv")
