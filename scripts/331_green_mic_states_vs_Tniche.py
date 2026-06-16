"""
331_green_mic_states_vs_Tniche.py
LITERATURE-ANCHORED microglial states: signatures derived from the Green2024 microglia reference
(green_mic_state_mean_expr.csv) — top per-state markers, restricted to our panel AND filtered for
microglial specificity (cross-lineage contamination genes removed). Only Green states resolvable on
our immune panel are kept. Score decontam microglia -> niche enrichment (Cliff's delta) across T subsets.
Green state -> consensus name (Paolicelli2022 nomenclature; Sun2023 / Green2024 identities):
  Mic.2=Homeostatic  Mic.9=MHC-II/APC  Mic.12=Activated-DAM  Mic.13=Lipid-DAM(AD)  Mic.15=Inflammatory/IEG  Mic.7=Phagocytic-myeloid
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
R=30.0; NTOP=10
# Green state -> (consensus name)
GREENNAME={"Mic.2":"Homeostatic (Mic.2)","Mic.9":"MHC-II/APC (Mic.9)","Mic.12":"Activated-DAM (Mic.12)",
           "Mic.13":"Lipid-DAM (Mic.13)","Mic.15":"Inflammatory/IEG (Mic.15)","Mic.7":"Phagocytic-myeloid (Mic.7)"}
KEEP=list(GREENNAME.keys())
# cross-lineage contamination genes to exclude from any microglial signature (T/NK/B/neuron/astro/oligo/vascular)
BLOCK=set("CD3D CD3E CD3G CD2 CD8A CD8B CD28 CD247 SKAP1 TBX21 GZMA GZMB GZMK GZMH NKG7 KLRD1 KLRB1 KLRF1 KLRC1 IL2RB IL2RG ICOS CTSW LAT2 LCK SH2D1A SH2D2A CCR9 LAG3 EOMES CD7 SLA2 FYB1 NFATC2 IKZF2 IKZF3 P2RX5 LILRB1 CD40LG XCL1 IL2RA CCL5 CD27 TC2N THEMIS ITK LIME1 "
            "CD19 MS4A1 CD79A CD79B JCHAIN BANK1 BLK BLNK TNFRSF13B TNFRSF13C IGHM IGSF6 FCRL2 FCMR MZB1 "
            "AQP4 GJA1 GFAP SLC1A2 SLC1A3 GLUL HOPX RORB BCAS4 "
            "MOG MAL PLP1 MOBP MBP UGT8 GLDN BCL11B CNR1 "
            "RBFOX3 GAD1 FOXP2 NELL2 RELN AUTS2 "
            "COL1A1 COL9A3 COL4A3 ACTA2 PDGFRB PECAM1 VCAN FBLN7 THBS1 "
            "FOXP3 GATA2 GATA3 KIT AGRP CD36 FABP4 JCHAIN MARCO MARCKSL1 CD14INVALID".split())
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
# ---- load Green ref mean expr, build panel+micro-filtered signatures ----
ae=pd.read_csv(NEW/"green_mic_state_mean_expr.csv",index_col=0)
ae.columns=[c.split(".",1)[-1] if c.startswith("SCT") else c for c in ae.columns]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
panel=[gn for gn in var if gn in ae.index]; M=ae.loc[panel,[c for c in ae.columns if c.startswith("Mic.")]]
Z=(M.sub(M.mean(1),axis=0)).div(M.std(1)+1e-9,axis=0)
SIG={}
for st in KEEP:
    cand=Z[st].sort_values(ascending=False)
    genes=[gn for gn in cand.index if gn not in BLOCK and M.loc[gn,st]>0.3][:NTOP]
    SIG[GREENNAME[st]]=genes
print("Literature-anchored (Green2024) microglial-state signatures on our panel:")
for k,v in SIG.items(): print(f"  {k:28}: {', '.join(v)}")
STATEORD=[GREENNAME[s] for s in KEEP]
# ---- score decontam microglia ----
is_mic=(v2=="Mic"); micidx=np.where(is_mic)[0]
A=ad.AnnData(X=Xd[micidx].copy(),obs=pd.DataFrame(index=idx[micidx]),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
for k,genes in SIG.items(): sc.tl.score_genes(A,[g for g in genes if g in A.var_names],score_name=k,ctrl_size=50)
# ---- proximity per T subset ----
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"]
run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object); labv=lab.reindex(idx).values
ent={}
for d in sorted([x for x in QC.iterdir() if x.is_dir()]):
    csv=d/"cell_metadata.csv"
    if not csv.exists(): continue
    df=pd.read_csv(csv,usecols=["EntityID","center_x","center_y"])
    for eid,x,y in zip(df["EntityID"].astype(str),df["center_x"],df["center_y"]): ent[(d.name,eid)]=(x,y)
mx=np.full(len(idx),np.nan); my=np.full(len(idx),np.nan)
for i,c in enumerate(idx):
    if "_" in c: pre,eid=c.rsplit("_",1); mx[i],my[i]=ent.get((pre,eid),(np.nan,np.nan))
hasxy=np.isfinite(mx); near={s:np.zeros(len(idx)) for s in SUBS}; anyT=np.zeros(len(idx))
for r in np.unique(run):
    ms=np.where(is_mic&(run==r)&hasxy)[0]
    if not len(ms): continue
    mxy=np.column_stack([mx[ms],my[ms]])
    for s in SUBS:
        ss=np.where((labv==s)&(run==r)&hasxy)[0]
        if len(ss): d,_=cKDTree(np.column_stack([mx[ss],my[ss]])).query(mxy,k=1); near[s][ms]=(d<=R).astype(int)
    for s in SUBS: anyT[ms]+=near[s][ms]
mh=hasxy[micidx]; nearM={s:(near[s][micidx]>=1)&mh for s in SUBS}; baseM=(anyT[micidx]==0)&mh
def cliffs(a,b):
    n1,n2=len(a),len(b); r=stats.rankdata(np.concatenate([a,b])); U=r[:n1].sum()-n1*(n1+1)/2; return 2*U/(n1*n2)-1
rows=[]
for s in SUBS:
    nm=nearM[s].values if hasattr(nearM[s],"values") else nearM[s]; ncell=int(nm.sum())
    for st in STATEORD:
        a=A.obs.loc[nm,st].values; b=A.obs.loc[baseM.values if hasattr(baseM,"values") else baseM,st].values
        if ncell>=5: d=cliffs(a,b); p=stats.mannwhitneyu(a,b).pvalue
        else: d,p=np.nan,np.nan
        rows.append(dict(subset=s,n_micnear=ncell,state=st,cliffs=d,p=p))
RES=pd.DataFrame(rows); ok=RES.p.notna(); pv=RES.loc[ok,"p"].values
o=np.argsort(pv); rk=np.empty(len(pv),int); rk[o]=np.arange(1,len(pv)+1); q=np.minimum.accumulate((pv*len(pv)/rk)[o][::-1])[::-1]
padj=np.full(len(pv),np.nan); padj[o]=np.minimum(q,1); RES.loc[ok,"padj"]=padj
RES.to_csv(NEW/"green_mic_state_enrichment_all_subsets.csv",index=False)
ncol={s:int((nearM[s].values if hasattr(nearM[s],'values') else nearM[s]).sum()) for s in SUBS}
print("\nmicroglia-near n per subset:",ncol,"  baseline:",int((baseM.values if hasattr(baseM,'values') else baseM).sum()))
piv=RES.pivot(index="state",columns="subset",values="cliffs").reindex(STATEORD)[SUBS]
pad=RES.pivot(index="state",columns="subset",values="padj").reindex(STATEORD)[SUBS]
print("\nCliff's delta (Green state x subset):"); print(piv.round(3).to_string())
# heatmap
fig,ax=plt.subplots(figsize=(10,4.2)); norm=TwoSlopeNorm(vmin=-0.2,vcenter=0,vmax=0.2)
im=ax.imshow(piv.values,cmap="RdBu_r",norm=norm,aspect="auto")
ax.set_xticks(range(len(SUBS))); ax.set_xticklabels([f"{s}\n(n={ncol[s]})" for s in SUBS],fontsize=8)
ax.set_yticks(range(len(STATEORD))); ax.set_yticklabels(STATEORD,fontsize=8.5)
for i in range(len(STATEORD)):
    for j in range(len(SUBS)):
        v=piv.values[i,j]; pq=pad.values[i,j]
        if np.isnan(v): ax.text(j,i,"–",ha="center",va="center",color="#999")
        elif pq<0.05: ax.text(j,i,"*",ha="center",va="center",fontsize=12,fontweight="bold")
fig.colorbar(im,ax=ax,shrink=0.7,label="Cliff's δ vs baseline\n(>0 state enriched near subset)")
ax.set_title("Green2024 microglial-state enrichment near each T subset\n(literature-anchored signatures, decontam counts; * BH-FDR<0.05; – n<5)",fontsize=10,fontweight="bold")
plt.tight_layout(); fig.savefig(NEW/"green_mic_state_enrichment.png",dpi=140,bbox_inches="tight"); plt.close()
pd.DataFrame({k:pd.Series(v) for k,v in SIG.items()}).to_csv(NEW/"green_mic_state_signatures.csv",index=False)
print("\nSaved: green_mic_state_enrichment.png + green_mic_state_enrichment_all_subsets.csv + green_mic_state_signatures.csv")
