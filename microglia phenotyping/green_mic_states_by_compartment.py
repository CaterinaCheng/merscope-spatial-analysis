"""
334_green_mic_states_by_compartment.py
Compare the literature-anchored Green2024 microglial STATES across vascular compartments
(perivascular <=30um / vessel-adjacent 30-100 / parenchymal >=100). Score decontam microglia
for each Green-state signature (from green_mic_state_signatures.csv), then:
  - mean z-scored state score per compartment (heatmap)
  - Cliff's delta vs parenchymal (peri & adj), BH-FDR (bar)
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy import stats
sc.settings.verbosity=0; plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
VESSEL=["End","Per","SMC"]
sig=pd.read_csv(NEW/"green_mic_state_signatures.csv")
SIG={c:[g for g in sig[c].dropna().tolist()] for c in sig.columns}
STATEORD=list(SIG.keys())
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
micidx=np.where(is_mic&hasxy&np.isfinite(dV))[0]
A=ad.AnnData(X=Xd[micidx].copy(),obs=pd.DataFrame(index=idx[micidx]),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
for k,genes in SIG.items(): sc.tl.score_genes(A,[g for g in genes if g in A.var_names],score_name=k,ctrl_size=50)
A.obs["comp"]=comp[micidx]
corder=["perivascular","vessel-adjacent","parenchymal"]
print("microglia per compartment:",A.obs["comp"].value_counts().reindex(corder).to_dict())
# RAW mean score_genes per compartment (background-subtracted; comparable across compartments)
S=A.obs[STATEORD]
Rc=S.groupby(A.obs["comp"]).mean().reindex(corder)[STATEORD].T  # states x comp, RAW
print("\nRAW mean signature score by compartment:"); print(Rc.round(3).to_string())
# Cliff's delta vs parenchymal
def cliffs(a,b):
    n1,n2=len(a),len(b); r=stats.rankdata(np.concatenate([a,b])); U=r[:n1].sum()-n1*(n1+1)/2; return 2*U/(n1*n2)-1
par=A.obs["comp"]=="parenchymal"; rows=[]
for st in STATEORD:
    b=A.obs.loc[par,st].values
    for cgrp in ["perivascular","vessel-adjacent"]:
        a=A.obs.loc[A.obs["comp"]==cgrp,st].values
        rows.append(dict(state=st,group=cgrp,cliffs=cliffs(a,b),p=stats.mannwhitneyu(a,b).pvalue))
RES=pd.DataFrame(rows); pv=RES.p.values; o=np.argsort(pv); rk=np.empty(len(pv),int); rk[o]=np.arange(1,len(pv)+1)
RES["padj"]=np.minimum(np.minimum.accumulate((pv*len(pv)/rk)[o][::-1])[::-1][np.argsort(o)],1)
RES.to_csv(NEW/"green_mic_state_by_compartment.csv",index=False)
print("\nCliff's delta vs parenchymal:"); print(RES.pivot(index="state",columns="group",values="cliffs").reindex(STATEORD).round(3).to_string())
# ONE-vs-REST per (state,compartment) for heatmap stars: compartment vs the other two combined
ovr=[]
for st in STATEORD:
    for c in corder:
        a=A.obs.loc[A.obs["comp"]==c,st].values; b=A.obs.loc[A.obs["comp"]!=c,st].values
        ovr.append(dict(state=st,comp=c,cliffs=cliffs(a,b),p=stats.mannwhitneyu(a,b).pvalue))
OVR=pd.DataFrame(ovr); pv2=OVR.p.values; o2=np.argsort(pv2); rk2=np.empty(len(pv2),int); rk2[o2]=np.arange(1,len(pv2)+1)
OVR["padj"]=np.minimum(np.minimum.accumulate((pv2*len(pv2)/rk2)[o2][::-1])[::-1][np.argsort(o2)],1)
OVRd=OVR.pivot(index="state",columns="comp",values="cliffs").reindex(STATEORD)[corder]
OVRp=OVR.pivot(index="state",columns="comp",values="padj").reindex(STATEORD)[corder]
print("\none-vs-rest Cliff's delta (heatmap stars):"); print(OVRd.round(3).to_string())

# ===== FIGURE =====
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,4.6),gridspec_kw={"width_ratios":[1,1.15]})
# heatmap RAW mean score; color = raw value on a single diverging scale centered at 0 (baseline)
vmax=np.abs(Rc.values).max(); norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
im=ax1.imshow(Rc.values,cmap="RdBu_r",norm=norm,aspect="auto")
ax1.set_xticks(range(3)); ax1.set_xticklabels([f"{c}\n(n={int((A.obs['comp']==c).sum())})" for c in corder],fontsize=8)
ax1.set_yticks(range(len(STATEORD))); ax1.set_yticklabels(STATEORD,fontsize=8.5)
for i in range(len(STATEORD)):
    for j in range(3):
        v=Rc.values[i,j]; star="*" if OVRp.values[i,j]<0.05 and abs(OVRd.values[i,j])>=0.1 else ""
        ax1.text(j,i,f"{v:.3f}{star}",ha="center",va="center",fontsize=8,color="white" if abs(v)>0.6*vmax else "black",fontweight="bold")
fig.colorbar(im,ax=ax1,shrink=0.7,label="RAW mean signature score\n(red>0, blue<0; 0 = baseline)")
ax1.set_title("Green microglial states by compartment (RAW mean score)\n(* = compartment vs other two: BH-FDR<0.05 & |δ|>=0.1)",fontsize=9,fontweight="bold")
# bar: Cliff's delta vs parenchymal
piv=RES.pivot(index="state",columns="group",values="cliffs").reindex(STATEORD); pad=RES.pivot(index="state",columns="group",values="padj").reindex(STATEORD)
y=np.arange(len(STATEORD)); w=0.38
for gname,off,c in [("perivascular",w/2,"#C0392B"),("vessel-adjacent",-w/2,"#E67E22")]:
    ax2.barh(y+off,piv[gname].values,w,color=c,edgecolor="#333",lw=0.3,label=gname)
    for yi,(v,pq) in zip(y+off,zip(piv[gname].values,pad[gname].values)):
        if pq<0.05: ax2.text(v+(0.005 if v>=0 else -0.005),yi,"*",va="center",ha="left" if v>=0 else "right",fontweight="bold")
ax2.set_yticks(y); ax2.set_yticklabels(STATEORD,fontsize=8.5); ax2.invert_yaxis(); ax2.axvline(0,color="#333",lw=0.7)
ax2.set_xlabel("Cliff's δ vs parenchymal (>0 enriched near vessels)"); ax2.legend(fontsize=8,loc="lower right")
ax2.set_title("Enrichment vs parenchymal microglia\n(* BH-FDR<0.05)",fontsize=9.5,fontweight="bold")
for sp in ("top","right","left"): ax2.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"green_mic_states_by_compartment.png",dpi=140,bbox_inches="tight"); plt.close()
print("\nSaved: green_mic_states_by_compartment.png + green_mic_state_by_compartment.csv")
