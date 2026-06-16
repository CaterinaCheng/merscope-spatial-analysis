"""
339_schpf_factor_mapping.py
Project our microglia onto the De Jager/Marshe MERSCOPE microglial scHPF FACTOR atlas
(Marshe 2025, PMC11974870; 23 retained factors). Reference = Table S3 gene loadings.
Because it's a FACTOR (continuous) framework (not discrete clusters), it scores well on our
panel. Build loading-anchored signatures (top loaded genes on our panel), score decontam
microglia, then: UMAP by dominant factor + key factor scores; compartment & T-subset enrichment.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy import stats
import warnings; warnings.filterwarnings("ignore")
sc.settings.verbosity=0; plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new"); REF=NEW/"reference"
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
VESSEL=["End","Per","SMC"]; MINGENES=5; TOPN=15
NAME={"scHPF_1":"IL/IFNg","scHPF_2":"Glycolysis","scHPF_3":"CIITAHigh","scHPF_4":"NPY1RHigh","scHPF_5":"Chemokine",
 "scHPF_7":"TLR/MAPK","scHPF_8":"OxPhos-1","scHPF_9":"Motility/adhesion","scHPF_10":"GRID2High","scHPF_11":"APOEHigh",
 "scHPF_14":"Stress","scHPF_15":"HLAHigh/APC","scHPF_16":"CX3CR1High","scHPF_17":"C1QHigh/phagocytic","scHPF_18":"OxPhos-2",
 "scHPF_19":"Senescence","scHPF_20":"IFN-I","scHPF_21":"OxPhos-3","scHPF_22":"PLCG2High","scHPF_23":"S100/TLR",
 "scHPF_24":"ActinFold","scHPF_25":"Immunoregulatory","scHPF_26":"GPNMBHigh"}
S3=pd.read_excel(REF/"PMC11974870/media-2.xlsx",sheet_name="Tab S3",header=3).dropna(subset=["gene"]).set_index("gene")
fac=[c for c in S3.columns if c.startswith("scHPF_") and "*" not in str(c)]  # asterisk marks excluded 6/12/13
L=S3[fac].apply(pd.to_numeric,errors="coerce")
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
vset=set(var)
# build loading-anchored signatures: top panel genes per factor by loading
SIG={}; cov={}
for c in fac:
    onpanel=[g for g in L[c].sort_values(ascending=False).index if g in vset]
    thr=0.15*L[c].max()
    genes=[g for g in onpanel if L.loc[g,c]>thr][:TOPN]
    cov[c]=len(genes)
    if len(genes)>=MINGENES: SIG[NAME[c]+f" ({c})"]=genes
print(f"shared atlas-panel genes: {len(vset & set(L.index))}")
print(f"SCOREABLE factors (>= {MINGENES} loaded panel genes): {len(SIG)} of 23")
for k,v in SIG.items(): print(f"  {k:28}: {', '.join(v)}")
notscore=[NAME[c]+f"({c})" for c in fac if cov[c]<MINGENES]; print("\nNOT scoreable (too few panel genes):",", ".join(notscore))
STATEORD=list(SIG.keys())
# score microglia
is_mic=(v2=="Mic"); is_ves=np.isin(v2,VESSEL); micidx=np.where(is_mic)[0]
A=ad.AnnData(X=Xd[micidx].copy(),obs=pd.DataFrame(index=idx[micidx]),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
for k,genes in SIG.items(): sc.tl.score_genes(A,[g for g in genes if g in A.var_names],score_name=k,ctrl_size=50)
Z=(A.obs[STATEORD]-A.obs[STATEORD].mean())/A.obs[STATEORD].std()
dom=Z.idxmax(1).values; dom[Z.max(1).values<0.1]="Mixed/low"; A.obs["factor"]=dom
print("\ndominant factor composition:",{k:round(100*(A.obs.factor==k).mean(),1) for k in STATEORD+['Mixed/low'] if (A.obs.factor==k).any()})
A.obs[STATEORD].to_csv(NEW/"microglia_scHPF_factor_scores.csv")
# compartment + T-subset proximity
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
hasxy=np.isfinite(mx); dV=np.full(len(idx),np.inf); near={s:np.zeros(len(idx)) for s in SUBS}; anyT=np.zeros(len(idx))
for r in np.unique(run):
    cs=np.where(is_mic&(run==r)&hasxy)[0]; vs=np.where(is_ves&(run==r)&hasxy)[0]
    if len(vs) and len(cs): dd,_=cKDTree(np.column_stack([mx[vs],my[vs]])).query(np.column_stack([mx[cs],my[cs]]),k=1); dV[cs]=dd
    ms=np.where(is_mic&(run==r)&hasxy)[0]
    if len(ms):
        mxy=np.column_stack([mx[ms],my[ms]])
        for s in SUBS:
            ss=np.where((labv==s)&(run==r)&hasxy)[0]
            if len(ss): d,_=cKDTree(np.column_stack([mx[ss],my[ss]])).query(mxy,k=1); near[s][ms]=(d<=30).astype(int)
        for s in SUBS: anyT[ms]+=near[s][ms]
comp=np.where(dV<=30,"perivascular",np.where(dV<100,"vessel-adjacent","parenchymal"))
A.obs["comp"]=comp[micidx]
mh=hasxy[micidx]; baseM=(anyT[micidx]==0)&mh
def cliffs(a,b):
    n1,n2=len(a),len(b); rk=stats.rankdata(np.concatenate([a,b])); U=rk[:n1].sum()-n1*(n1+1)/2; return 2*U/(n1*n2)-1
# compartment enrichment (peri & adj vs paren)
par=A.obs.comp.values=="parenchymal"; rowsC=[]
for st in STATEORD:
    b=A.obs.loc[par,st].values
    for cg in ["perivascular","vessel-adjacent"]:
        a=A.obs.loc[A.obs.comp.values==cg,st].values; rowsC.append(dict(factor=st,grp=cg,cliffs=cliffs(a,b),p=stats.mannwhitneyu(a,b).pvalue))
RC=pd.DataFrame(rowsC)
# T-subset enrichment (near subset vs baseline)
rowsT=[]
for s in SUBS:
    nm=(near[s][micidx]>=1)&mh; ncell=int(nm.sum())
    for st in STATEORD:
        if ncell>=5: a=A.obs.loc[nm,st].values; b=A.obs.loc[baseM,st].values; d=cliffs(a,b); p=stats.mannwhitneyu(a,b).pvalue
        else: d,p=np.nan,np.nan
        rowsT.append(dict(subset=s,n=ncell,factor=st,cliffs=d,p=p))
RT=pd.DataFrame(rowsT)
for R in (RC,RT):
    ok=R.p.notna(); pv=R.loc[ok,"p"].values; o=np.argsort(pv); rk=np.empty(len(pv),int); rk[o]=np.arange(1,len(pv)+1)
    R.loc[ok,"padj"]=np.minimum(np.minimum.accumulate((pv*len(pv)/rk)[o][::-1])[::-1][np.argsort(o)],1)
RC.to_csv(NEW/"scHPF_factor_compartment.csv",index=False); RT.to_csv(NEW/"scHPF_factor_Tsubset.csv",index=False)

# UMAP coords from 336
co=pd.read_csv(NEW/"microglia_umap_coords.csv",index_col=0); U=co.reindex(A.obs_names)[["umap1","umap2"]].values
KEY=[k for k in STATEORD if any(t in k for t in ["HLAHigh","GPNMBHigh","CX3CR1High","Chemokine","C1QHigh","APOEHigh"])]
fig=plt.figure(figsize=(18,9)); gs=fig.add_gridspec(2,4,height_ratios=[1.3,1])
axA=fig.add_subplot(gs[0,0:2]); order=STATEORD+["Mixed/low"]; pal=cm.get_cmap("tab20",len(order))
for i,st in enumerate(order):
    m=A.obs.factor.values==st
    if m.sum(): axA.scatter(U[m,0],U[m,1],s=2.5,c=[pal(i)],label=f"{st} ({100*m.mean():.0f}%)",linewidths=0,rasterized=True)
axA.set_title("Microglia UMAP — dominant scHPF factor (Marshe2025 MERSCOPE atlas)",fontsize=10.5,fontweight="bold")
axA.set_xticks([]); axA.set_yticks([]); axA.legend(markerscale=4,fontsize=7,loc="center left",bbox_to_anchor=(1.0,0.5),frameon=False)
for sp in axA.spines.values(): sp.set_visible(False)
axB=fig.add_subplot(gs[0,2:4])
for c in ["parenchymal","vessel-adjacent","perivascular"]:
    m=A.obs.comp.values==c
    axB.scatter(U[m,0],U[m,1],s=2.5,c={"perivascular":"#C0392B","vessel-adjacent":"#E67E22","parenchymal":"#2471A3"}[c],label=c,linewidths=0,rasterized=True)
axB.set_title("Vascular compartment",fontsize=10.5,fontweight="bold"); axB.set_xticks([]); axB.set_yticks([]); axB.legend(markerscale=4,fontsize=8,frameon=False)
for sp in axB.spines.values(): sp.set_visible(False)
for i,st in enumerate(KEY[:4]):
    ax=fig.add_subplot(gs[1,i]); sv=A.obs[st].values; vmax=np.percentile(np.abs(sv),98)
    s=ax.scatter(U[:,0],U[:,1],s=1.5,c=sv,cmap="RdBu_r",vmin=-vmax,vmax=vmax,linewidths=0,rasterized=True)
    ax.set_title(st,fontsize=8.5,fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"microglia_umap_scHPF_factors.png",dpi=150,bbox_inches="tight"); plt.close()

# compartment + T-subset enrichment heatmaps
fig2,(ax1,ax2)=plt.subplots(1,2,figsize=(17,6),gridspec_kw={"width_ratios":[0.5,1]})
pivC=RC.pivot(index="factor",columns="grp",values="cliffs").reindex(STATEORD); padC=RC.pivot(index="factor",columns="grp",values="padj").reindex(STATEORD)
im1=ax1.imshow(pivC[["perivascular","vessel-adjacent"]].values,cmap="RdBu_r",norm=TwoSlopeNorm(0,vmin=-0.2,vmax=0.2),aspect="auto")
ax1.set_xticks([0,1]); ax1.set_xticklabels(["perivascular","vessel-adj"],fontsize=8); ax1.set_yticks(range(len(STATEORD))); ax1.set_yticklabels(STATEORD,fontsize=8)
for i in range(len(STATEORD)):
    for j,gg in enumerate(["perivascular","vessel-adjacent"]):
        if padC.values[i,j]<0.05 and abs(pivC[gg].values[i])>=0.1: ax1.text(j,i,"*",ha="center",va="center",fontweight="bold")
ax1.set_title("scHPF factor enrichment\nvs parenchymal (*FDR<.05,|δ|≥.1)",fontsize=9,fontweight="bold"); fig2.colorbar(im1,ax=ax1,shrink=0.6,label="Cliff's δ")
pivT=RT.pivot(index="factor",columns="subset",values="cliffs").reindex(STATEORD)[SUBS]; padT=RT.pivot(index="factor",columns="subset",values="padj").reindex(STATEORD)[SUBS]
nsub={s:int(RT[RT.subset==s].n.iloc[0]) for s in SUBS}
im2=ax2.imshow(pivT.values,cmap="RdBu_r",norm=TwoSlopeNorm(0,vmin=-0.2,vmax=0.2),aspect="auto")
ax2.set_xticks(range(len(SUBS))); ax2.set_xticklabels([f"{s}\n(n={nsub[s]})" for s in SUBS],fontsize=7.5); ax2.set_yticks(range(len(STATEORD))); ax2.set_yticklabels(STATEORD,fontsize=8)
for i in range(len(STATEORD)):
    for j in range(len(SUBS)):
        v=pivT.values[i,j]
        if np.isnan(v): ax2.text(j,i,"–",ha="center",va="center",color="#999")
        elif padT.values[i,j]<0.05: ax2.text(j,i,"*",ha="center",va="center",fontweight="bold")
ax2.set_title("scHPF factor enrichment near each T subset (vs baseline microglia; *FDR<.05)",fontsize=9,fontweight="bold"); fig2.colorbar(im2,ax=ax2,shrink=0.6,label="Cliff's δ")
plt.tight_layout(); fig2.savefig(NEW/"scHPF_factor_enrichment.png",dpi=150,bbox_inches="tight"); plt.close()
print("\nSaved: microglia_umap_scHPF_factors.png + scHPF_factor_enrichment.png + factor scores/enrichment csvs")
