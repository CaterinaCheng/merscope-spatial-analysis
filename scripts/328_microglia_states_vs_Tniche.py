"""
328_microglia_states_vs_Tniche.py
Define a microglial STATE taxonomy (not one hand-picked module) and map T-cell proximity onto it.
- Input: ambient-decontaminated counts (counts_decontam) for microglia only.
- Leiden subcluster -> annotate each cluster by resolvable state signatures
  (Homeostatic / DAM-ARM / MHC-II-APC / Phagocytic / Inflammatory; IFN & proliferating NOT on panel).
- Map niche: microglia in T niche (>=1 T/NK <=30um), near CD8 TRM1, near CD8 TRM2, vs baseline.
- Outputs: UMAP of microglial states; state composition baseline vs niche vs near-TRM2;
  continuous signature-score enrichment (niche vs baseline, near-TRM2 vs baseline).
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
R=30.0
SIG={
 "Homeostatic":["CX3CR1","TMEM119","CSF1R","MARCKS","C1QA","C1QB","C1QC"],
 "DAM/ARM":["APOE","GPNMB","ITGAX","TREM2","TYROBP","LPL","CST7","CLEC7A","LGALS3","FABP5","NUPR1","TIMP2"],
 "MHC-II/APC":["CD74","HLA-DRA","HLA-DRB1","HLA-DPA1","HLA-DQB1","CIITA","HLA-DMB"],
 "Phagocytic":["CD68","CTSB","GRN","FCGR3A","MSR1","CD163","MRC1"],
 "Inflammatory":["IL1B","CCL2","CCL3","CCL4","CCL5","KLF4"],
}
STATEORD=list(SIG.keys())
SCOL={"Homeostatic":"#3498DB","DAM/ARM":"#E74C3C","MHC-II/APC":"#9B59B6","Phagocytic":"#E67E22","Inflammatory":"#F1C40F","Mixed/low":"#95A5A6"}
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
is_mic=(v2=="Mic"); micidx=np.where(is_mic)[0]; print("microglia:",len(micidx))
# microglia AnnData on decontam counts
A=ad.AnnData(X=Xd[micidx].copy(),obs=pd.DataFrame(index=idx[micidx]),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A); A.raw=A
# restrict clustering features to genes detected in >=5% microglia (avoid sparse off-lineage noise)
det=np.asarray((A.X>0).mean(0)).ravel(); keep=np.array(var)[det>=0.05]
Asc=A[:,keep].copy(); sc.pp.scale(Asc,max_value=10); sc.tl.pca(Asc,n_comps=30)
sc.pp.neighbors(Asc,n_neighbors=15,n_pcs=30); sc.tl.leiden(Asc,resolution=0.6,key_added="leiden")
sc.tl.umap(Asc); A.obs["leiden"]=Asc.obs["leiden"].values; A.obsm["X_umap"]=Asc.obsm["X_umap"]
# signature scores
for st,gs in SIG.items(): sc.tl.score_genes(A,[g for g in gs if g in A.var_names],score_name=st,ctrl_size=50)
S=A.obs[STATEORD].copy(); Z=(S-S.mean())/S.std()  # z across microglia
# annotate each leiden cluster -> dominant state by mean z (label Mixed/low if max mean z < 0.15)
cl_mean=Z.groupby(A.obs["leiden"].values).mean()
cl_state={}
for cl,row in cl_mean.iterrows():
    cl_state[cl]="Mixed/low" if row.max()<0.15 else row.idxmax()
A.obs["state"]=A.obs["leiden"].map(cl_state).values
print("\nLeiden cluster -> state (mean z of each signature):"); print(cl_mean.round(2).to_string())
print("\ncluster sizes:",A.obs["leiden"].value_counts().sort_index().to_dict())
print("state assignment:",{cl:cl_state[cl] for cl in cl_mean.index})
print("\nstate composition (all microglia):"); print((100*A.obs["state"].value_counts(normalize=True)).round(1).to_string())

# ===== spatial niche membership =====
run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object); labv=lab.reindex(idx).values
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"]
isT=np.isin(labv,SUBS)
ent={}
for d in sorted([x for x in QC.iterdir() if x.is_dir()]):
    csv=d/"cell_metadata.csv"
    if not csv.exists(): continue
    df=pd.read_csv(csv,usecols=["EntityID","center_x","center_y"])
    for eid,x,y in zip(df["EntityID"].astype(str),df["center_x"],df["center_y"]): ent[(d.name,eid)]=(x,y)
mx=np.full(len(idx),np.nan); my=np.full(len(idx),np.nan)
for i,c in enumerate(idx):
    if "_" in c: pre,eid=c.rsplit("_",1); mx[i],my[i]=ent.get((pre,eid),(np.nan,np.nan))
hasxy=np.isfinite(mx)
nT=np.zeros(len(idx)); nTRM1=np.zeros(len(idx)); nTRM2=np.zeros(len(idx))
for r in np.unique(run):
    ms=np.where(is_mic&(run==r)&hasxy)[0]
    if not len(ms): continue
    mxy=np.column_stack([mx[ms],my[ms]])
    for arr,msk in [(nT,isT),(nTRM1,labv=="CD8 TRM 1"),(nTRM2,labv=="CD8 TRM 2")]:
        ss=np.where(msk&(run==r)&hasxy)[0]
        if len(ss): d,_=cKDTree(np.column_stack([mx[ss],my[ss]])).query(mxy,k=1); arr[ms]=(d<=R).astype(int)
mo=pd.Series(index=idx,dtype=object);
micmask=pd.Series(is_mic&hasxy,index=idx)
A.obs["in_Tniche"]=nT[micidx]>=1; A.obs["near_TRM1"]=nTRM1[micidx]>=1; A.obs["near_TRM2"]=nTRM2[micidx]>=1
A.obs["hasxy"]=hasxy[micidx]
base=A.obs["hasxy"]&(~A.obs["in_Tniche"])
print(f"\nmicroglia: baseline(no T)={int(base.sum())}  in T niche={int((A.obs.hasxy&A.obs.in_Tniche).sum())}  near TRM2={int((A.obs.hasxy&A.obs.near_TRM2).sum())}")

# ===== composition: baseline vs T-niche vs near-TRM2 =====
def comp(mask):
    vc=A.obs.loc[mask,"state"].value_counts(normalize=True)*100; return vc.reindex(STATEORD+["Mixed/low"]).fillna(0)
groups={"Baseline\n(no T/NK)":base,"In T niche":A.obs.hasxy&A.obs.in_Tniche,"Near CD8 TRM1":A.obs.hasxy&A.obs.near_TRM1,"Near CD8 TRM2":A.obs.hasxy&A.obs.near_TRM2}
C=pd.DataFrame({k:comp(v) for k,v in groups.items()})
print("\nstate composition (%) by niche:"); print(C.round(1).to_string())

# ===== continuous enrichment: signature score niche vs baseline =====
print("\nsignature-score enrichment (Cliff's delta & MWU p), niche/near-TRM2 vs baseline:")
def cliffs(a,b):
    # P(a>b)-P(a<b) via rank
    n1,n2=len(a),len(b); r=stats.rankdata(np.concatenate([a,b])); R1=r[:n1].sum()
    U=R1-n1*(n1+1)/2; return 2*U/(n1*n2)-1
enr=[]
for st in STATEORD:
    for gname,gm in [("T niche",A.obs.hasxy&A.obs.in_Tniche),("near TRM2",A.obs.hasxy&A.obs.near_TRM2)]:
        a=A.obs.loc[gm,st].values; b=A.obs.loc[base,st].values
        d=cliffs(a,b); p=stats.mannwhitneyu(a,b).pvalue; enr.append(dict(state=st,group=gname,cliffs=d,p=p))
ENR=pd.DataFrame(enr); print(ENR.round(3).to_string(index=False)); ENR.to_csv(NEW/"microglia_state_enrichment.csv",index=False)

# ===== FIGURE =====
fig=plt.figure(figsize=(17,5.4));
ax0=fig.add_subplot(1,3,1)
for st in STATEORD+["Mixed/low"]:
    m=A.obs["state"].values==st
    ax0.scatter(A.obsm["X_umap"][m,0],A.obsm["X_umap"][m,1],s=2,c=SCOL[st],label=f"{st} ({100*m.mean():.0f}%)",linewidths=0,rasterized=True)
ax0.set_title("Microglial states (Leiden on decontam counts)",fontsize=9.5,fontweight="bold"); ax0.set_xticks([]); ax0.set_yticks([])
ax0.legend(markerscale=4,fontsize=7,loc="upper right",frameon=False)
for sp in ax0.spines.values(): sp.set_visible(False)
ax1=fig.add_subplot(1,3,2)
bottom=np.zeros(len(C.columns))
for st in STATEORD+["Mixed/low"]:
    ax1.bar(range(len(C.columns)),C.loc[st].values,bottom=bottom,color=SCOL[st],edgecolor="white",lw=0.5,label=st)
    for j,(v,b) in enumerate(zip(C.loc[st].values,bottom)):
        if v>=5: ax1.text(j,b+v/2,f"{v:.0f}",ha="center",va="center",fontsize=7,color="white",fontweight="bold")
    bottom+=C.loc[st].values
ax1.set_xticks(range(len(C.columns))); ax1.set_xticklabels(C.columns,fontsize=8); ax1.set_ylabel("% of microglia"); ax1.set_ylim(0,100)
ax1.set_title("State composition by T proximity",fontsize=9.5,fontweight="bold")
for sp in ("top","right"): ax1.spines[sp].set_visible(False)
ax2=fig.add_subplot(1,3,3)
piv=ENR.pivot(index="state",columns="group",values="cliffs").reindex(STATEORD)
pp=ENR.pivot(index="state",columns="group",values="p").reindex(STATEORD)
y=np.arange(len(STATEORD)); w=0.38
for k,(gname,off,c) in enumerate([("T niche",w/2,"#16A085"),("near TRM2",-w/2,"#C0392B")]):
    ax2.barh(y+off,piv[gname].values,w,color=c,edgecolor="#333",lw=0.3,label=gname)
    for yi,(v,pv) in zip(y+off,zip(piv[gname].values,pp[gname].values)):
        if pv<0.05: ax2.text(v+(0.01 if v>=0 else -0.01),yi,"*",va="center",ha="left" if v>=0 else "right",fontweight="bold")
ax2.set_yticks(y); ax2.set_yticklabels(STATEORD,fontsize=8); ax2.invert_yaxis(); ax2.axvline(0,color="#333",lw=0.7)
ax2.set_xlabel("Cliff's delta vs baseline (>0 enriched)"); ax2.set_title("Signature-score enrichment vs baseline\n(* MWU p<0.05)",fontsize=9.5,fontweight="bold"); ax2.legend(fontsize=8)
for sp in ("top","right","left"): ax2.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"microglia_states_vs_Tniche.png",dpi=140,bbox_inches="tight"); plt.close()
C.to_csv(NEW/"microglia_state_composition_by_niche.csv")
A.obs[["leiden","state","in_Tniche","near_TRM1","near_TRM2"]].to_csv(NEW/"microglia_state_assignment.csv")
print("\nSaved: microglia_states_vs_Tniche.png + 3 csvs")
